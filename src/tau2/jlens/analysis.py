"""Build Neuronpedia-style, all-position J-Lens views from tau2 traces.

The analyzer consumes the exact prompt and completion token IDs written by
``InstrumentedHFBackend``.  It never re-renders the conversation and never
truncates a recorded call.  Every recorded token position is read out at the
selected fitted layers and at the model's actual final layer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from tau2.jlens.profiles import PROFILES, JLensModelProfile, get_profile

SUPPORTED_SCHEMAS = frozenset({"tau2-jlens-v1", "tau2-jlens-v2"})


@dataclass(frozen=True)
class PositionSegment:
    """A semantic overlay in the exact request-plus-response token sequence."""

    label: str
    kind: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        """Return the shape expected by the interactive viewer."""
        return {
            "label": self.label,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
        }


class ExactInputModel:
    """LensModel adapter that replays recorded token IDs without truncation."""

    def __init__(self, model: Any, token_ids: Sequence[int], torch: Any) -> None:
        self._model = model
        self._token_ids = tuple(int(token_id) for token_id in token_ids)
        self._torch = torch
        self.tokenizer = model.tokenizer
        self.n_layers = model.n_layers
        self.d_model = model.d_model
        self.layers = model.layers

    def encode(self, _text: str, *, max_length: int = 512) -> Any:
        """Return the immutable recorded IDs and refuse implicit truncation."""
        if len(self._token_ids) > max_length:
            raise ValueError(
                f"recorded call has {len(self._token_ids)} tokens, exceeding "
                f"--max-seq-len={max_length}; no tokens were truncated"
            )
        return self._torch.tensor(
            [self._token_ids],
            dtype=self._torch.long,
            device=self._model.input_device,
        )

    def forward(self, input_ids: Any) -> Any:
        return self._model.forward(input_ids)

    def unembed(self, residual: Any) -> Any:
        return self._model.unembed(residual)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash an ordered token sequence using the trace writer's encoding."""
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _discover_trace_files(values: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            discovered.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            discovered.append(path)
        else:
            raise FileNotFoundError(path)
    files = sorted(set(discovered))
    if not files:
        raise ValueError("no J-Lens JSONL trace files were found")
    return files


def validate_record(record: dict[str, Any], *, source: str = "trace") -> None:
    """Validate the exact-token portion of a telemetry record."""
    schema = record.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported schema {schema!r} in {source}")
    for key in ("record_id", "task_id", "turn_index", "input_ids", "generated_ids"):
        if key not in record:
            raise ValueError(f"trace record missing {key!r} in {source}")
    input_ids = record["input_ids"]
    generated_ids = record["generated_ids"]
    if not isinstance(input_ids, list) or not all(
        isinstance(value, int) for value in input_ids
    ):
        raise ValueError(f"input_ids is not a flat integer list in {source}")
    if not isinstance(generated_ids, list) or not all(
        isinstance(value, int) for value in generated_ids
    ):
        raise ValueError(f"generated_ids is not a flat integer list in {source}")
    expected_input_hash = record.get("input_ids_sha256")
    if expected_input_hash and token_ids_sha256(input_ids) != expected_input_hash:
        raise ValueError(f"input token hash mismatch in {source}")
    expected_generated_hash = record.get("generated_ids_sha256")
    if expected_generated_hash and token_ids_sha256(generated_ids) != expected_generated_hash:
        raise ValueError(f"generated token hash mismatch in {source}")
    expected_full_hash = record.get("full_ids_sha256")
    if expected_full_hash and token_ids_sha256([*input_ids, *generated_ids]) != expected_full_hash:
        raise ValueError(f"full token hash mismatch in {source}")


def load_records(values: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load, validate, deduplicate, and deterministically order trace records."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _discover_trace_files(values):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            source = f"{path}:{line_number}"
            validate_record(record, source=source)
            record_id = str(record["record_id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            record["_trace_path"] = str(path)
            records.append(record)
    records.sort(
        key=lambda item: (
            str(item.get("task_id", "")),
            int(item.get("turn_index", -1)),
            str(item.get("record_id", "")),
        )
    )
    if not records:
        raise ValueError("J-Lens trace files contained no records")
    return records


def record_token_ids(record: dict[str, Any]) -> tuple[int, ...]:
    """Return the exact request-plus-response token sequence."""
    return tuple(int(value) for value in record["input_ids"]) + tuple(
        int(value) for value in record["generated_ids"]
    )


def semantic_positions(record: dict[str, Any]) -> dict[str, list[int]]:
    """Read semantic prediction positions from v2 or legacy observer data."""
    value = record.get("semantic_positions")
    if not isinstance(value, dict):
        measurement = record.get("measurement") or {}
        value = measurement.get("positions") if isinstance(measurement, dict) else {}
    output: dict[str, list[int]] = {}
    if isinstance(value, dict):
        for label, positions in value.items():
            if isinstance(positions, list):
                output[str(label)] = [
                    int(position)
                    for position in positions
                    if isinstance(position, int)
                ]
    return output


def semantic_segments(record: dict[str, Any]) -> list[PositionSegment]:
    """Build exact request, response, boundary, tool-name, and argument overlays."""
    prompt_length = len(record["input_ids"])
    total_length = prompt_length + len(record["generated_ids"])
    segments: list[PositionSegment] = []
    if prompt_length:
        segments.append(PositionSegment("request context + tool schema", "system", 0, prompt_length))
    if total_length > prompt_length:
        segments.append(
            PositionSegment(
                "current assistant response",
                "assistant",
                prompt_length,
                total_length,
            )
        )
    for boundary in record.get("boundaries") or []:
        if prompt_length:
            segments.append(
                PositionSegment(
                    str(boundary).replace("_", " "),
                    f"boundary-{boundary}",
                    prompt_length - 1,
                    prompt_length,
                )
            )
    for label, positions in semantic_positions(record).items():
        valid = sorted({position for position in positions if 0 <= position < total_length})
        if not valid:
            continue
        if label == "initial_decision":
            start, end, kind = valid[0], valid[-1] + 1, "boundary-decision"
        else:
            start = max(0, valid[0] + 1)
            end = min(total_length, valid[-1] + 2)
            kind = "tool-name" if label.endswith("_name") else "arguments"
        if end > start:
            segments.append(
                PositionSegment(label.replace("_", " "), kind, start, end)
            )
    unique = {
        (segment.label, segment.kind, segment.start, segment.end): segment
        for segment in segments
    }
    return sorted(
        unique.values(),
        key=lambda segment: (segment.start, -(segment.end - segment.start), segment.kind),
    )


def pinned_token_ids(record: dict[str, Any], token_ids: Sequence[int]) -> set[int]:
    """Pin generated tool/argument tokens for full rank tracking."""
    pins: set[int] = set()
    for label, positions in semantic_positions(record).items():
        if label == "initial_decision":
            continue
        for position in positions:
            target = position + 1
            if 0 <= target < len(token_ids):
                pins.add(int(token_ids[target]))
    if not pins:
        prompt_length = len(record["input_ids"])
        pins.update(int(token_id) for token_id in token_ids[prompt_length:])
    return pins


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return cleaned or "item"


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def position_readout_rows(
    record: dict[str, Any],
    slice_data: Any,
    tokenizer: Any,
) -> Iterable[dict[str, Any]]:
    """Yield one compact row for every analyzed position and layer."""
    full_ids = record_token_ids(record)
    segment_values = semantic_segments(record)
    tracked = {
        int(token_id): index
        for index, token_id in enumerate(slice_data.tracked_token_ids)
    }
    for local_position in range(slice_data.seq_len):
        position = slice_data.ctx_offset + local_position
        segment = next(
            (
                item
                for item in sorted(
                    (
                        candidate
                        for candidate in segment_values
                        if candidate.start <= position < candidate.end
                    ),
                    key=lambda candidate: candidate.end - candidate.start,
                )
            ),
            None,
        )
        next_token_id = int(full_ids[position + 1]) if position + 1 < len(full_ids) else None
        next_track = tracked.get(next_token_id) if next_token_id is not None else None
        for layer_index, layer in enumerate(slice_data.layers):
            top_ids = [
                int(value) for value in slice_data.top_ids[local_position, layer_index]
            ]
            top_ranks = [
                int(value) for value in slice_data.top_ranks[local_position, layer_index]
            ]
            next_rank = (
                int(slice_data.rank_tensor[local_position, layer_index, next_track])
                if next_track is not None
                else None
            )
            yield {
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "turn_index": record["turn_index"],
                "position": position,
                "source_token_id": int(full_ids[position]),
                "source_token": _decode_token(tokenizer, int(full_ids[position])),
                "next_token_id": next_token_id,
                "next_token": (
                    _decode_token(tokenizer, next_token_id)
                    if next_token_id is not None
                    else None
                ),
                "next_token_rank": next_rank,
                "layer": layer,
                "segment": segment.label if segment else None,
                "segment_kind": segment.kind if segment else None,
                "top_token_ids": json.dumps(top_ids),
                "top_tokens": json.dumps(
                    [_decode_token(tokenizer, token_id) for token_id in top_ids],
                    ensure_ascii=False,
                ),
                "top_ranks": json.dumps(top_ranks),
            }


POSITION_FIELDS = [
    "record_id",
    "task_id",
    "turn_index",
    "position",
    "source_token_id",
    "source_token",
    "next_token_id",
    "next_token",
    "next_token_rank",
    "layer",
    "segment",
    "segment_kind",
    "top_token_ids",
    "top_tokens",
    "top_ranks",
]


def _write_position_rows(
    path: Path, rows: Iterable[dict[str, Any]], *, append: bool
) -> None:
    mode = "at" if append else "wt"
    with gzip.open(path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSITION_FIELDS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    if requested != "auto":
        value = getattr(torch, requested, None)
        if value is None:
            raise ValueError(f"unknown torch dtype {requested!r}")
        return value
    if device.startswith("cuda"):
        major, _minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_model_and_lens(
    profile: JLensModelProfile,
    *,
    device_name: str,
    dtype_name: str,
    attention: str,
    allow_low_vram: bool,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load the pinned model and lens after CUDA/VRAM preflight."""
    try:
        import jlens
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install tau2 with the jlens extra before running full analysis"
        ) from exc
    device = device_name
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu: dict[str, Any] = {"device": device}
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 2**30
        if free_gib < profile.min_vram_gib and not allow_low_vram:
            raise RuntimeError(
                f"{profile.name} expects about {profile.min_vram_gib:.0f} GiB free "
                f"VRAM; only {free_gib:.1f} GiB is free"
            )
        major, minor = torch.cuda.get_device_capability()
        gpu.update(
            {
                "name": torch.cuda.get_device_name(),
                "capability": f"{major}.{minor}",
                "free_vram_gib": round(free_gib, 2),
                "total_vram_gib": round(total_bytes / 2**30, 2),
            }
        )
    dtype = _resolve_dtype(torch, dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.model_id, revision=profile.model_revision
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.model_revision,
        dtype=dtype,
        attn_implementation=attention,
        low_cpu_mem_usage=True,
    ).to(device)
    hf_model.eval()
    model = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.from_pretrained(
        profile.lens_repo,
        filename=profile.lens_file,
        revision=profile.lens_revision,
    )
    if lens.d_model != model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} does not match model d_model={model.d_model}"
        )
    if not lens.source_layers:
        raise ValueError("the fitted lens has no source layers")
    gpu.update(
        {
            "dtype": str(dtype).replace("torch.", ""),
            "model_d_model": model.d_model,
            "lens_d_model": lens.d_model,
        }
    )
    return torch, tokenizer, model, lens, gpu


def _validate_record_profile(
    record: dict[str, Any], profile: JLensModelProfile
) -> None:
    model = record.get("model") or {}
    recorded_id = model.get("name_or_path") if isinstance(model, dict) else None
    if recorded_id and recorded_id != profile.model_id:
        raise ValueError(
            f"trace model {recorded_id!r} does not match profile model "
            f"{profile.model_id!r}"
        )
    resolved = model.get("resolved_revision") if isinstance(model, dict) else None
    if resolved and resolved != profile.model_revision:
        raise ValueError(
            f"trace revision {resolved!r} does not match profile revision "
            f"{profile.model_revision!r}"
        )


def _relative_view_path(record: dict[str, Any]) -> Path:
    return Path(
        "views",
        f"task-{_safe_name(record['task_id'])}",
        f"turn-{int(record['turn_index'])}",
        _safe_name(record["record_id"]),
    )


def analyze_record(
    record: dict[str, Any],
    *,
    output_dir: Path,
    profile: JLensModelProfile,
    torch: Any,
    tokenizer: Any,
    model: Any,
    lens: Any,
    top_k: int,
    layer_stride: int,
    position_chunk_size: int,
    max_seq_len: int,
    max_tracked: int | None,
) -> tuple[dict[str, Any], Iterable[dict[str, Any]]]:
    """Analyze one exact trace and write its interactive viewer artifacts."""
    from jlens.vis import build_page, compute_slice

    _validate_record_profile(record, profile)
    token_ids = record_token_ids(record)
    if len(token_ids) > max_seq_len:
        raise ValueError(
            f"record {record['record_id']} contains {len(token_ids)} tokens, "
            f"exceeding --max-seq-len={max_seq_len}; no tokens were truncated"
        )
    if len(token_ids) > profile.max_context_tokens:
        raise ValueError(
            f"record {record['record_id']} exceeds the pinned model context limit "
            f"of {profile.max_context_tokens} tokens"
        )
    display_text = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    pins = pinned_token_ids(record, token_ids)
    exact_model = ExactInputModel(model, token_ids, torch)
    slice_data = compute_slice(
        exact_model,
        lens,
        display_text,
        top_n=top_k,
        max_tracked=max_tracked,
        pinned_token_ids=pins,
        layer_stride=layer_stride,
        last_n_tokens=None,
        max_seq_len=max_seq_len,
        mask_display=False,
        position_chunk_size=position_chunk_size,
    )
    if slice_data.seq_len != len(token_ids):
        raise AssertionError(
            f"viewer retained {slice_data.seq_len}/{len(token_ids)} positions"
        )
    if model.n_layers - 1 not in slice_data.layers:
        raise AssertionError("viewer does not include the actual model-final layer")
    segments = [segment.to_dict() for segment in semantic_segments(record)]
    relative = _relative_view_path(record)
    view_dir = output_dir / relative
    view_dir.mkdir(parents=True, exist_ok=True)
    page, _raw_bytes, _payload_bytes = build_page(
        slice_data,
        display_text,
        title=(
            f"tau2 J-Space · task {record['task_id']} · turn {record['turn_index']}"
        ),
        description=(
            f"{profile.name} · exact trace · {len(token_ids)} positions × "
            f"{len(slice_data.layers)} layers · top-{top_k}"
        ),
        pinned_token_ids=pins,
        mode="fetch",
        out_dir=view_dir,
        position_segments=segments,
    )
    (view_dir / "index.html").write_text(page, encoding="utf-8")
    raw_bytes = (
        slice_data.top_ids.nbytes
        + slice_data.top_ranks.nbytes
        + slice_data.rank_tensor.nbytes
    )
    payload_bytes = sum(path.stat().st_size for path in view_dir.rglob("*.bin"))
    entry = {
        "record_id": record["record_id"],
        "task_id": record["task_id"],
        "turn_index": record["turn_index"],
        "mode": record.get("mode"),
        "model": profile.name,
        "token_source": "recorded-token-ids",
        "prompt_tokens": len(record["input_ids"]),
        "completion_tokens": len(record["generated_ids"]),
        "total_tokens": len(token_ids),
        "rendered_positions": slice_data.seq_len,
        "all_positions": True,
        "layers": slice_data.layers,
        "top_k": top_k,
        "position_chunk_size": position_chunk_size,
        "tracked_tokens": len(slice_data.tracked_token_ids),
        "segments": segments,
        "raw_grid_bytes": raw_bytes,
        "compressed_grid_bytes": payload_bytes,
        "trace_path": record.get("_trace_path"),
        "href": (relative / "index.html").as_posix(),
        "status": "ok",
    }
    (view_dir / "analysis.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return entry, position_readout_rows(record, slice_data, tokenizer)


def write_index(output_dir: Path, entries: list[dict[str, Any]]) -> Path:
    """Write the filterable catalog for all generated J-Space views."""
    data = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>tau2 Full-Position J-Lens</title><style>
body{font:14px system-ui;margin:24px;color:#172033;background:#f6f8fb}h1{margin-bottom:4px}
.muted{color:#667085}.filters{display:flex;gap:8px;margin:18px 0;flex-wrap:wrap}
select{padding:7px;border:1px solid #ccd1da;border-radius:7px;background:white}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left}
th{position:sticky;top:0;background:#eef2f7}a{color:#155eef}.error{color:#b42318}
</style><h1>tau2 Full-Position J-Lens</h1>
<div class="muted">Neuronpedia-style J-Space views over every exact prompt and completion position.</div>
<div class="muted" id="count"></div><div class="filters" id="filters"></div>
<table><thead><tr><th>task</th><th>turn</th><th>model</th><th>positions</th><th>layers</th><th>top-K</th><th>status</th><th>J-Space</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=__DATA__,fields=['task_id','turn_index','model','status'],chosen={};
const filters=document.getElementById('filters');for(const f of fields){const s=document.createElement('select');s.innerHTML='<option value="">all '+f+'</option>'+[...new Set(data.map(x=>String(x[f]??'')))].sort().map(v=>`<option>${v}</option>`).join('');s.onchange=()=>{chosen[f]=s.value;render()};filters.appendChild(s)}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const shown=data.filter(x=>fields.every(f=>!chosen[f]||String(x[f]??'')===chosen[f]));document.getElementById('count').textContent=`${shown.length} call(s)`;document.getElementById('rows').innerHTML=shown.map(x=>`<tr><td>${esc(x.task_id)}</td><td>${esc(x.turn_index)}</td><td>${esc(x.model)}</td><td>${esc(x.rendered_positions??'')}</td><td>${Array.isArray(x.layers)?x.layers.length:''}</td><td>${esc(x.top_k??'')}</td><td class="${x.status==='error'?'error':''}">${esc(x.status)}</td><td>${x.href?`<a href="${encodeURI(x.href)}">open</a>`:esc(x.error||'—')}</td></tr>`).join('')}render();</script></html>"""
    path = output_dir / "index.html"
    path.write_text(template.replace("__DATA__", data), encoding="utf-8")
    return path


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add the ``tau2 jlens`` command arguments."""
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help="J-Lens JSONL files or directories containing them",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="qwen3.5-4b"
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--turns", nargs="+", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--position-chunk-size", type=int, default=128)
    parser.add_argument("--max-tracked", type=int)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--allow-low-vram", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")


def run_from_args(args: argparse.Namespace) -> Path:
    """Run inspection or all-position analysis from parsed CLI arguments."""
    if args.top_k < 1 or args.layer_stride < 1:
        raise ValueError("--top-k and --layer-stride must be positive")
    if args.position_chunk_size < 1 or args.max_seq_len < 1:
        raise ValueError("--position-chunk-size and --max-seq-len must be positive")
    if args.max_tracked is not None and args.max_tracked < 0:
        raise ValueError("--max-tracked must be non-negative")
    profile = get_profile(args.profile)
    records = load_records(args.traces)
    if args.task_ids:
        task_ids = {str(value) for value in args.task_ids}
        records = [record for record in records if str(record["task_id"]) in task_ids]
    if args.turns:
        turns = set(args.turns)
        records = [record for record in records if int(record["turn_index"]) in turns]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise ValueError("trace selection is empty")
    for record in records:
        _validate_record_profile(record, profile)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path("data/jlens_analysis").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "Tau2FullPositionJLensManifestV1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "inspect_only" if args.inspect_only else "running",
        "profile": profile.to_dict(),
        "all_positions": True,
        "top_k": args.top_k,
        "layer_stride": args.layer_stride,
        "position_chunk_size": args.position_chunk_size,
        "max_tracked": args.max_tracked,
        "max_seq_len": args.max_seq_len,
        "selected_records": [
            {
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "turn_index": record["turn_index"],
                "prompt_tokens": len(record["input_ids"]),
                "completion_tokens": len(record["generated_ids"]),
                "trace_path": record.get("_trace_path"),
            }
            for record in records
        ],
        "entries": [],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.inspect_only:
        write_index(output_dir, [])
        return output_dir

    torch, tokenizer, model, lens, runtime = load_model_and_lens(
        profile,
        device_name=args.device,
        dtype_name=args.dtype,
        attention=args.attn_implementation,
        allow_low_vram=args.allow_low_vram,
    )
    manifest["runtime"] = runtime
    entries: list[dict[str, Any]] = []
    position_path = output_dir / "position_readouts.csv.gz"
    wrote_rows = False
    for index, record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] task={record['task_id']} "
            f"turn={record['turn_index']} record={record['record_id']}",
            flush=True,
        )
        try:
            entry, rows = analyze_record(
                record,
                output_dir=output_dir,
                profile=profile,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                lens=lens,
                top_k=args.top_k,
                layer_stride=args.layer_stride,
                position_chunk_size=args.position_chunk_size,
                max_seq_len=args.max_seq_len,
                max_tracked=args.max_tracked,
            )
            _write_position_rows(position_path, rows, append=wrote_rows)
            wrote_rows = True
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as exc:
            entry = {
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "turn_index": record["turn_index"],
                "model": profile.name,
                "status": "error",
                "error": str(exc),
            }
        entries.append(entry)
        manifest["entries"] = entries
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    error_count = sum(entry["status"] == "error" for entry in entries)
    index_path = write_index(output_dir, entries)
    manifest.update(
        {
            "status": "complete_with_errors" if error_count else "complete",
            "outputs": {
                "index": str(index_path.resolve()),
                "position_readouts": str(position_path.resolve()) if wrote_rows else None,
            },
            "summary": {"calls": len(entries), "errors": error_count},
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if error_count and not args.allow_errors:
        raise RuntimeError(
            f"J-Lens analysis failed for {error_count}/{len(entries)} calls; "
            f"inspect {manifest_path}"
        )
    return output_dir


def main(argv: list[str] | None = None) -> Path:
    """Standalone entry point used by tests and direct module execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    main()
