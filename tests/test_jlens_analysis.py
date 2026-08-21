import argparse
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tau2.jlens.analysis import (
    POSITION_FIELDS,
    analyze_record,
    configure_parser,
    load_records,
    pinned_token_ids,
    position_readout_rows,
    record_token_ids,
    run_from_args,
    semantic_segments,
    token_ids_sha256,
)
from tau2.jlens.profiles import get_profile


def _record(**updates):
    input_ids = [10, 11, 12]
    generated_ids = [20, 21, 22]
    value = {
        "schema_version": "tau2-jlens-v2",
        "record_id": "record-1",
        "task_id": "task-a",
        "turn_index": 2,
        "input_ids": input_ids,
        "input_ids_sha256": token_ids_sha256(input_ids),
        "generated_ids": generated_ids,
        "generated_ids_sha256": token_ids_sha256(generated_ids),
        "full_ids_sha256": token_ids_sha256([*input_ids, *generated_ids]),
        "boundaries": ["after_tool_result"],
        "semantic_positions": {
            "initial_decision": [2],
            "tool_0_name": [2, 3],
            "tool_0_arguments": [4],
        },
    }
    value.update(updates)
    return value


def test_trace_loader_validates_hashes_deduplicates_and_sorts(tmp_path):
    first = _record(record_id="b", task_id="z", turn_index=1)
    second = _record(record_id="a", task_id="a", turn_index=0)
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(json.dumps(value) for value in [first, second, first]) + "\n",
        encoding="utf-8",
    )

    records = load_records([tmp_path])

    assert [record["record_id"] for record in records] == ["a", "b"]
    assert all(record["_trace_path"] == str(path.resolve()) for record in records)

    tampered = _record(generated_ids=[99])
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated token hash mismatch"):
        load_records([path])


def test_semantic_overlays_and_pins_use_exact_absolute_positions():
    record = _record()
    token_ids = record_token_ids(record)

    segments = semantic_segments(record)

    assert len(token_ids) == 6
    assert any(
        segment.kind == "tool-name" and (segment.start, segment.end) == (3, 5)
        for segment in segments
    )
    assert pinned_token_ids(record, token_ids) == {20, 21, 22}


def test_position_export_has_one_row_for_every_position_and_layer():
    record = _record()
    token_ids = record_token_ids(record)
    seq_len = len(token_ids)
    layers = [0, 4, 7]
    top_ids = np.empty((seq_len, len(layers), 2), dtype=np.int32)
    top_ids[..., 0] = 30
    top_ids[..., 1] = 31
    top_ranks = np.zeros_like(top_ids)
    top_ranks[..., 1] = 1
    tracked = [20, 21, 22, 30, 31]
    rank_tensor = np.zeros((seq_len, len(layers), len(tracked)), dtype=np.int32)
    slice_data = SimpleNamespace(
        seq_len=seq_len,
        ctx_offset=0,
        layers=layers,
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor,
    )

    class Tokenizer:
        @staticmethod
        def decode(ids, **_kwargs):
            return f"<{ids[0]}>"

    rows = list(position_readout_rows(record, slice_data, Tokenizer()))

    assert len(rows) == seq_len * len(layers)
    assert set(rows[0]) == set(POSITION_FIELDS)
    assert {row["position"] for row in rows} == set(range(seq_len))
    assert {row["layer"] for row in rows} == set(layers)


def test_cli_defaults_to_unwindowed_all_position_analysis():
    parser = argparse.ArgumentParser()
    configure_parser(parser)

    args = parser.parse_args(["trace.jsonl"])

    assert args.layer_stride == 4
    assert args.position_chunk_size == 128
    assert args.max_seq_len == 32768
    assert args.profile == "qwen3.5-4b"


def test_qwen35_9b_base_profile_is_pinned_for_diagnostic_analysis():
    profile = get_profile("qwen3.5-9b-base")

    assert profile.model_id == "Qwen/Qwen3.5-9B-Base"
    assert profile.model_revision == "68c46c4b3498877f3ef123c856ecfde50c39f404"
    assert profile.lens_file.endswith("Qwen3.5-9B-Base_jacobian_lens.pt")


def test_analyze_record_builds_fetch_view_for_every_exact_position(tmp_path):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        @staticmethod
        def decode(ids, **_kwargs):
            return "".join(f"<{value}>" for value in ids)

    class Model:
        n_layers = 2
        d_model = 4
        layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        input_device = torch.device("cpu")
        tokenizer = Tokenizer()
        weight = torch.arange(40 * 4, dtype=torch.float32).reshape(40, 4) / 100

        @classmethod
        def forward(cls, input_ids):
            hidden = torch.nn.functional.one_hot(
                input_ids % cls.d_model, num_classes=cls.d_model
            ).float()
            for layer in cls.layers:
                hidden = layer(hidden)
            return hidden

        @classmethod
        def unembed(cls, residual):
            return residual @ cls.weight.T

    class Lens:
        d_model = 4
        source_layers = [0]
        jacobians = {0: torch.eye(4)}

        @staticmethod
        def transport(residual, _layer):
            return residual

    entry, rows = analyze_record(
        _record(),
        output_dir=tmp_path,
        profile=get_profile("qwen3.5-4b"),
        torch=torch,
        tokenizer=Model.tokenizer,
        model=Model(),
        lens=Lens(),
        top_k=2,
        layer_stride=1,
        position_chunk_size=2,
        max_seq_len=32,
        max_tracked=None,
    )
    rows = list(rows)

    assert entry["all_positions"] is True
    assert entry["rendered_positions"] == len(record_token_ids(_record()))
    assert entry["layers"] == [0, 1]
    assert len(rows) == entry["rendered_positions"] * len(entry["layers"])
    view = tmp_path / entry["href"]
    assert view.is_file()
    assert (view.parent / "meta.json").is_file()
    assert (view.parent / "slice.bin").is_file()


def test_inspect_only_cli_writes_manifest_without_loading_model(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    output = tmp_path / "inspection"

    result = run_from_args(
        parser.parse_args(
            [str(trace), "--inspect-only", "--output-dir", str(output)]
        )
    )

    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "inspect_only"
    assert manifest["all_positions"] is True
    assert manifest["selected_records"][0]["record_id"] == "record-1"


def test_gzip_position_export_is_standard_csv(tmp_path):
    path = tmp_path / "rows.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("position,layer\n0,0\n")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert handle.read().splitlines() == ["position,layer", "0,0"]
