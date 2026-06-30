# FastWAM LIBERO Deep Exemplar

`fastwam-libero` is the first deep EazyWAM exemplar: it shows the complete
public product path for one real WAM across model discovery, asset preparation,
one-observation inference, serving smoke, native eval, explicit reference eval,
trace inspection, comparison, and optimization profiles.

It is not EazyWAM's architecture center. Future model entries should reach the
same maturity by filling in the same checklist with their own backend,
processor, workloads, profiles, evidence, and known gaps. They should not copy
FastWAM internals into core harness contracts.

This page does not change FastWAM's core acceleration code and does not create a
new speedup claim. For setup and current evidence, see
`docs/fastwam_libero_eval_setup.md`. For acceleration status and measured
evidence rules, see `docs/acceleration_methods.md` and
`docs/optimization_profiles.md`.

## Reusable Exemplar Checklist

| Capability | Exemplar requirement | `fastwam-libero` status |
|---|---|---|
| Model library | `wam list` and `wam info <model-id>` identify the curated entry, source, assets, defaults, supported profiles, and known gaps. | Present as `fastwam-libero`. |
| Prepare | `wam prepare <model-id>` can create the cache layout and fetch pullable declared assets without installing the runtime environment. | `--asset eval` covers the released checkpoint, dataset stats, Wan VAE, T5 encoder, and tokenizer files. |
| Doctor | `wam doctor <model-id>` checks cache and runtime readiness without modifying the environment. | Used before native run, serve, and eval paths. |
| Run | `wam run <model-id> --input obs.json` runs one explicit observation through the native product path. | Native path uses the vendored FastWAM runtime and does not require an upstream checkout. |
| Serve | `wam serve <model-id>` starts a resident policy process, and `--smoke --smoke-input` performs a job-local health and inference check. | Serve smoke with a real observation has been verified on the documented SuperPod H800 path. |
| Product eval | `wam eval <model-id> --workload <workload>` wraps the product backend and processor inside a workload or simulator loop. | `libero-single-task` is the default native LIBERO eval workload with documented acceptance evidence. |
| Reference eval | Official upstream scripts are available only through explicit reference mode. | `wam eval fastwam-libero --reference ...` is for parity and debugging, not the default product eval path. |
| Trace | Runs, serve requests, native evals, reference evals, and optimization profiles emit JSONL trace events. | Trace docs define completion events and profile metadata expectations. |
| Compare | Baseline and variant traces can be compared with runtime, output, latency, memory, and eval-metric gates. | `wam compare` supports run, serve, native eval, and external eval summaries when traces include comparable data. |
| Optimization profiles | At least one profile can be enabled through shared `--opt` controls, with status and evidence kept explicit. | `scheduler` is the documented FastWAM eval profile; `cuda_graph` has measured evidence in its recorded scope; `teacache` remains experimental. |
| Status language | The model entry names verified paths and known gaps without claiming general parity or blanket speedups. | Native/reference parity is not statistically established; smaller-GPU VRAM floor and long-running serve stability remain gaps. |

## Tutorial: Full Product Path

The commands below assume a prepared FastWAM-compatible runtime, either from the
self-managed path or the container path in `docs/fastwam_libero_eval_setup.md`.

Set shared paths:

```bash
export WAM_CACHE_DIR=/path/to/wam-cache
export WAM_RUN_ROOT=/path/to/runs/fastwam-libero-tutorial
mkdir -p "$WAM_CACHE_DIR" "$WAM_RUN_ROOT"
```

### 1. Inspect The Model Entry

```bash
wam list
wam info fastwam-libero
wam doctor fastwam-libero --cache-dir "$WAM_CACHE_DIR"
```

Expected result: `wam info` shows the model entry, processor, backend path,
default action shape, supported optimization profiles, and known gaps.
`wam doctor` reports missing assets or runtime requirements before model load.

### 2. Prepare Assets

```bash
wam prepare fastwam-libero \
  --cache-dir "$WAM_CACHE_DIR" \
  --download \
  --asset eval
```

Expected result: the cache contains the released policy checkpoint, dataset
statistics, Wan VAE, T5 encoder, and tokenizer files documented in
`docs/fastwam_libero_eval_setup.md`. `prepare` does not install CUDA, MuJoCo,
LIBERO, containers, or cluster launchers.

### 3. Run One Explicit Observation

```bash
wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output "$WAM_RUN_ROOT/run-action.json" \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir "$WAM_RUN_ROOT/run-baseline"
```

Expected result: the output JSON includes an action result and trace path. The
trace should finish with `run_end.status="ok"` and include an action chunk
shape compatible with the model entry, currently `[32, 7]` for FastWAM LIBERO.
This path uses an explicit observation and should not fall back to synthetic
smoke input.

### 4. Smoke-Test Job-Local Serving

```bash
wam serve fastwam-libero \
  --smoke \
  --smoke-input examples/fastwam_libero/obs.json \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir "$WAM_RUN_ROOT/serve-smoke" \
  --smoke-timeout 300
```

Expected result: the command starts a local resident server, checks health,
posts the observation to `/infer`, prints the response, and exits. The trace
should contain `serve_start`, `serve_ready`, `serve_request_start`,
`serve_request_end`, and `backend_close`.

For a long-running job-local server, omit `--smoke` and send requests to
`POST /infer` with the observation contract described in
`docs/cli_entrypoints.md`.

### 5. Run Native Product Eval

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir "$WAM_RUN_ROOT/eval-native" \
  --summary-path "$WAM_RUN_ROOT/eval-native-summary.json"
```

Expected result: this uses the harness-owned native LIBERO single-task loop. A
passing product-path trace should contain `native_eval_end`, should not contain
`external_eval_plan`, and should finish with `run_end.status="ok"`.

### 6. Keep Reference Eval Explicit

Reference evals call upstream official FastWAM scripts for parity or debugging.
They are not the default EazyWAM product path and should not be used as product
eval evidence unless the task explicitly asks for reference evidence.

```bash
wam eval fastwam-libero \
  --reference \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir "$WAM_CACHE_DIR" \
  --upstream-dir /path/to/FastWAM \
  --trace-dir "$WAM_RUN_ROOT/eval-reference" \
  --summary-path "$WAM_RUN_ROOT/eval-reference-summary.json"
```

Expected result: the trace records `external_eval_plan` and
`external_eval_end`. That is useful for parity checks, but it is intentionally
separate from the native product eval path.

### 7. Run An Optimization Profile

Use the shared `--opt` surface rather than model-specific public flags:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --opt scheduler \
  --set num_inference_steps=6 \
  --set sigma_shift=3.0 \
  --cache-dir "$WAM_CACHE_DIR" \
  --trace-dir "$WAM_RUN_ROOT/eval-scheduler" \
  --summary-path "$WAM_RUN_ROOT/eval-scheduler-summary.json"
```

Expected result: the trace shows the requested `scheduler` profile and backend
metadata such as `scheduler_name`, `num_inference_steps`, `sigma_shift`,
`schedule_type`, and `denoise_wall_ms` when reported by the native backend.

Do not promote this command alone as a speedup. A speedup claim requires a
comparable baseline, variant trace, output gate, eval metric gate when relevant,
fallback status, and hardware/runtime scope as described in
`docs/acceleration_methods.md`.

### 8. Inspect Traces

```bash
find "$WAM_RUN_ROOT" -name trace.jsonl -print
```

For native eval evidence, verify that the product trace has `native_eval_end`
and lacks `external_eval_plan`. For reference eval evidence, verify that
`external_eval_plan` and `external_eval_end` are present.

### 9. Compare Baseline And Variant

```bash
BASELINE_TRACE=$(find "$WAM_RUN_ROOT/eval-native" -name trace.jsonl -print -quit)
VARIANT_TRACE=$(find "$WAM_RUN_ROOT/eval-scheduler" -name trace.jsonl -print -quit)

wam compare "$BASELINE_TRACE" "$VARIANT_TRACE" --max-action-drift 0.001
```

Expected result: `wam compare` prints a JSON summary with latency or wall-time
statistics, runtime contract gate status, output gate status, optimization
profile differences, backend metadata summaries, and eval metrics when the
traces include them.

Treat `invalid` or `not_comparable` as validation output, not as a tool crash.
EazyWAM should not report a speedup when action shape, runtime contract, output
drift, fallback status, or eval metric evidence is missing or failing.

## Current Evidence Boundaries

- `fastwam-libero` is the first real deep exemplar, not a privileged
  architecture center.
- Native `run`, `serve`, and `eval` use the FastWAM runtime vendored in
  EazyWAM; a FastWAM upstream checkout is only needed for explicit reference
  eval parity checks.
- Native-vs-reference parity is not statistically established.
- Long-running serve stability and a smaller-GPU VRAM floor are not yet proven.
- `scheduler` and `cuda_graph` have measured evidence only within their
  documented FastWAM scopes. `teacache` is experimental, and `vla_cache` is
  unsupported for FastWAM because it targets OpenVLA-family models.

If a future tutorial or ticket needs stronger benchmark claims, model status,
or parity evidence than the current docs support, record the gap and keep the
ticket out of `Done` until the evidence exists.
