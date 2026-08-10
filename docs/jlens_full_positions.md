# Full-position J-Lens analysis

tau2 can capture the exact token IDs used for each local Hugging Face agent
turn and replay those IDs through the Jacobian Lens viewer. The resulting
J-Space pages follow the same interaction pattern as Neuronpedia: a
position-by-layer heatmap, selectable cells, token pinning, rank heatmaps,
rank-by-layer and rank-by-position plots, and per-cell top-token tables.

## Windows: setup through viewer in one command

From PowerShell, the bundled setup script installs `uv` when needed, installs
Python 3.12 and project dependencies, runs the selected tasks, captures exact
token traces, analyzes every position, starts a local server, and opens the
viewer:

```powershell
Set-Location D:\jlens\tau2-bench
powershell -ExecutionPolicy Bypass -File scripts\setup_and_run_jlens.ps1 `
  -Start 165 `
  -Count 1 `
  -Profile qwen3.5-4b
```

The combined one-based catalog is `1..50` airline, `51..164` retail, and
`165..2449` telecom. Use `-ListOnly` to validate a selection without loading a
model and `-NoBrowser` to leave the viewer URL in the terminal without opening
it. Qwen3.5-9B-Base is available as `-Profile qwen3.5-9b-base`, but it is a
pre-trained-only checkpoint and its output is diagnostic rather than a valid
instruction/tool-use benchmark result.

Ranges containing airline or retail require an OpenAI-compatible user-model
endpoint. Override its defaults with `-UserModel` and `-UserApiBase`. Telecom
is run in direct solo mode and does not require that endpoint.

## Linux/Vast.ai: progress-friendly airline and retail runs

The Linux runner prints the active phase plus elapsed time, GPU utilization,
trace count, and completed-view count every 15 seconds. It prompts for the
OpenAI user-simulator key without echoing it and never starts an HTTP server.

Run the two-task boundary pilot first:

```bash
bash scripts/setup_and_run_jlens.sh --pilot
```

To run the first two catalog positions instead (airline task IDs 0 and 1),
override the pilot range at invocation time:

```bash
bash scripts/setup_and_run_jlens.sh --pilot --start 1 --count 2
```

Then run all 50 airline and 114 retail tasks:

```bash
bash scripts/setup_and_run_jlens.sh --full
```

Trace paths are stable so re-running the same mode can auto-resume generation.
Each analysis attempt uses a new result directory, preventing an old error
manifest from being overwritten.

The analyzer never reconstructs a prompt from log text. It uses the recorded
`input_ids + generated_ids`, validates their hashes, and refuses to truncate a
trace. By default it includes every token position, every fourth fitted lens
layer, the last fitted layer, and the model's actual final layer. Use
`--layer-stride 1` to include every fitted layer.

## 1. Capture exact traces

Install the J-Lens extra from the repository root:

```bash
uv sync --extra jlens --extra dev
```

For a normal user-agent tau2 run:

```bash
uv run tau2 run \
  --domain airline \
  --agent jlens_hf_agent \
  --agent-llm Qwen/Qwen3.5-4B \
  --agent-llm-args '{
    "hf_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "jlens_mode": "off",
    "jlens_telemetry_path": "data/jlens_traces/{task_id}/{simulation_id}.jsonl",
    "hf_dtype": "bfloat16",
    "max_new_tokens": 256,
    "do_sample": false
  }' \
  --task-ids 0
```

For an existing no-user/solo task, replace the agent with
`jlens_direct_solo`. The `off` mode here means “do not collect the small online
observer sample”; exact-token trace capture still occurs. This is the preferred
capture mode before exhaustive offline analysis because it avoids an extra
teacher-forced pass during the benchmark.

The trace path supports both `{task_id}` and `{simulation_id}` placeholders, so
parallel trials do not append unrelated calls to the same file.

### Select one range across all three domains

`scripts/run_jlens_range.py` treats the task sets as one stable, one-based
catalog in this order: airline, retail, telecom. It accepts a starting position
and an exact count, including ranges that cross a domain boundary:

```bash
uv run python scripts/run_jlens_range.py --start 45 --count 10 --profile qwen3-8b
```

Use `--list-only` to print the total/domain counts and exact selected task IDs
without loading the model. Airline and retail use `jlens_hf_agent` with the
configured user simulator; telecom automatically uses `jlens_direct_solo`.
The user simulator defaults to `openai/qwen3:8b` at
`http://127.0.0.1:11434/v1` and can be changed with `--user-model` and
`--user-api-base`.

## 2. Inspect before allocating the model

```bash
uv run tau2 jlens data/jlens_traces \
  --profile qwen3.5-4b \
  --output-dir data/jlens_analysis \
  --inspect-only
```

This validates JSONL syntax, schemas, exact-token hashes, deduplicates records,
applies filters, and writes a manifest without loading PyTorch, the model, or
the fitted lens.

## 3. Analyze every position

```bash
uv run tau2 jlens data/jlens_traces \
  --profile qwen3.5-4b \
  --output-dir data/jlens_analysis \
  --top-k 10 \
  --layer-stride 1 \
  --position-chunk-size 128
```

For Qwen3-8B traces use `--profile qwen3-8b`. The
`--profile qwen3.6-27b` option uses the same Qwen3.6-27B family shown in the
Neuronpedia reference and requires roughly 70 GiB of free VRAM. Profiles pin
both the base model revision and its fitted lens revision; a trace from another
model or resolved revision is rejected rather than silently analyzed with
mismatched weights.

`--profile qwen3.5-9b-base` is also available with the pinned 9B Base lens.
That checkpoint is pre-trained-only, so its results are diagnostic rather than
a valid instruction-following or tool-use benchmark result.

`--position-chunk-size` limits the peak vocabulary-logit allocation while
preserving every token position. Reduce it when VRAM is tight. The analyzer
performs a VRAM preflight; `--allow-low-vram` acknowledges an intentional
low-memory attempt but does not enable truncation.

## 4. Open the result

The viewer uses fetch sidecars, so serve the output directory over HTTP:

```bash
cd data/jlens_analysis
python -m http.server 8000
```

Open `http://localhost:8000/`. The root page filters calls by task, turn, model,
and status. Each call links to an interactive J-Space page.

The output also contains:

- `manifest.json`: pinned provenance, selected calls, runtime/device details,
  per-call status, layer set, and size statistics.
- `position_readouts.csv.gz`: one row per analyzed `(call, position, layer)`
  with the exact source token, following token, semantic segment, top token IDs,
  decoded top tokens, and ranks.
- `views/task-*/turn-*/<record-id>/analysis.json`: per-call metadata.
- `views/.../meta.json`, `slice.bin`, and `ranks/*.bin`: compact viewer data.

## Reproducibility contract

- Generation and analysis share the exact recorded token IDs.
- `input_ids`, `generated_ids`, and their concatenation are hash-checked.
- Model and lens revisions are pinned by the selected profile.
- Every recorded position is asserted present in the rendered grid.
- The actual model-final layer is asserted present even when it has no fitted
  transport (the identity/final-logit readout is used).
- Errors are recorded per call, and the command exits nonzero unless
  `--allow-errors` is explicitly set.
