# J-Lens τ²-Telecom plumbing gate

This integration treats τ²-Telecom as the primary mechanistic benchmark. The
first milestone is deliberately narrower than a benchmark run: it checks the
shared local Hugging Face backend on three Telecom `small` tasks in No-User
mode.

It must not be reported as a τ² score, used for calibration, or pooled with an
Agents' Last Exam result.

## Environment

```powershell
uv sync --python 3.12 --extra dev --extra jlens
$env:PYTHONUTF8 = "1"
uv run tau2 check-data
```

The `jlens` extra installs the sibling `../jacobian-lens` checkout in editable
mode. Result manifests pin the upstream source revisions:

- Jacobian Lens: `581d398`
- τ²: `1d244f5`
- Agents' Last Exam: `d332c1a`

On Windows, the default package index may resolve a CPU-only PyTorch wheel.
For an NVIDIA host that supports CUDA 12.6, install the official CUDA wheel
after `uv sync`:

```powershell
uv pip install --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu126 `
  --reinstall "torch==2.13.0+cu126"
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Use the version and CUDA index appropriate for the host, as listed in the
[official PyTorch installation guide](https://docs.pytorch.org/get-started/locally/).

## Run the gate

```powershell
uv run python scripts/run_jlens_telecom_smoke.py `
  --model Qwen/Qwen3-0.6B `
  --revision c1899de289a04d12100db370d81485cdf75e47ca `
  --device auto `
  --sdpa-backend efficient `
  --dtype float16 `
  --max-input-tokens 16384 `
  --max-new-tokens 64
```

The default model revision is pinned to the immutable commit shown above.
When changing `--model`, pass its matching immutable `--revision`. The loaded
model and tokenizer commit hashes are also captured from the objects and
written to `manifest.json`.

The script runs the same cached model bundle twice:

1. `off`: deterministic baseline generation with exact input-token capture.
2. `observe`: identical generation followed by a teacher-forced residual-only
   measurement pass.

It exits non-zero unless actions and exact prompt IDs match turn by turn,
observer measurements exist, and strict environment replay reproduces both
rewards.

## Artifacts

Artifacts are written to `data/jlens_smoke/`, which is gitignored:

- `manifest.json`: requested and resolved revisions, task IDs, source state,
  run options, and OS/Python/PyTorch/CUDA/GPU runtime details.
- `off/<task>/jlens.jsonl`: baseline token and action trace.
- `observe/<task>/jlens.jsonl`: the same trace plus selected residuals and
  teacher-forced motorization measurements.
- `*/<task>/simulation.json`: regular τ² trajectory and reward.
- `smoke_report.json`: parity and replay gates for all three tasks.

Every JSONL row stores the exact generation `input_ids`, its SHA-256 hash, the
pre-truncation token count, the used token count, generated IDs, parsed tool
calls, and semantic boundaries. Only the first and last residual at each
selected semantic span are persisted; full-sequence activations are not.

## Mechanistic definition

For target token \(w\) and fitted layer \(l\), the source-space concept
direction is:

\[
v_{l,w} =
\frac{J_l^\top W_U[w]}
{\lVert J_l^\top W_U[w] \rVert_2}.
\]

`finite_difference_token_effect` checks that a positive local step along this
direction increases the intended token's unembedded logit. The default unit
suite includes a finite-difference regression test.

## Leakage boundary

The online prompt is constructed only from the public domain policy, the
ticket, allowed trajectory history, and tool schemas. Evaluation actions,
environment assertions, hidden database state, graders, and validity labels
are never consulted by the backend or controller. They remain offline
evaluation inputs.

The registered first-milestone factory is `jlens_direct_solo`. Plan, Verify,
Plan+Verify, budget-matched, Default, and Oracle-Plan variants should only be
added after this parity gate and the independent Telecom validity oracle pass.
