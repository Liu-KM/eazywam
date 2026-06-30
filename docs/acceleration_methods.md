# Acceleration Method Catalog

EazyWAM treats acceleration as a first-class product surface alongside the
model library. The model library answers "what WAM can I run?" The acceleration
method catalog answers "what runtime method can I enable, where does it apply,
and what evidence supports it?"

## Method Versus Profile

An acceleration method is the backend-integrated implementation: source code,
adapter code, hook point, runtime dependency, and model-specific behavior.

An optimization profile is the EazyWAM control and validation contract for that
method. The profile gives the method a stable name, parameters, compatibility
rules, trace fields, fallback behavior, and compare/eval expectations. Public
enablement should use the shared product controls:

```bash
wam run <model-id> --input obs.json --opt <method>
wam eval <model-id> --workload <workload> --opt <method>
wam serve <model-id> --opt <method>
```

Method-specific parameters should use shared overrides or request
`runtime_options`, not new model-specific public flags.

## Status Labels

Use these labels for acceleration method status:

| Status | Meaning |
|---|---|
| `planned` | The method is in scope, but there is no stable EazyWAM enablement path yet. |
| `implemented` | Code and a profile path exist, but the method is not yet validated as a reportable speedup. |
| `experimental` | Code exists, but quality gates, fallbacks, runtime stability, or scope are not mature enough for speedup claims. |
| `measured` | The method has the minimum evidence below and can be described as a proven speedup only for the recorded model/workload/runtime scope. |
| `unsupported` | The method is known not to apply to the model, backend, workload, device, or runtime path. |

Status is scoped. A method can be `measured` for one model/workload and
`unsupported` or `planned` for another.

## Minimum Measured Evidence

A `measured` method needs an evidence record with:

- model id and workload.
- baseline command.
- variant command with `--opt <method>` or the equivalent profile declaration.
- trace path, report path, or summary artifact.
- latency or memory comparison.
- action shape gate and output drift gate when applicable.
- simulator success, episode metrics, or task metric when relevant.
- fallback status and fallback reason.
- hardware, device, dtype, runtime, and backend environment.
- statement of scope, such as task id, seed, trial count, and whether the
  result applies only to a single-task smoke or a broader suite.

Observed latency gains without a passing output gate, success metric, or
trace-visible profile metadata may be recorded as audit data, but not promoted
as `measured`.

Use `docs/acceleration_validation_runbook.md` for the public Acceptance
Validation and Measured Validation loop, the Evidence Summary template, and the
Evidence Bundle v0.1 example.

## Current Catalog

| Method | Profile family | Enablement | Applicability | Status | Evidence and notes |
|---|---|---|---|---|---|
| `action_chunk_scheduling` | `action_runtime` | manifest defaults and action horizon/replan controls | FastWAM, Cosmos-Policy, DreamZero entries declare action-chunk behavior where supported | `implemented` | Compatibility/runtime behavior; not a standalone speedup claim. |
| `scheduler` | `scheduler` | `--opt scheduler`, `--set num_inference_steps=...`, `--set sigma_shift=...` | FastWAM LIBERO and RoboTwin native eval paths | `measured` for documented FastWAM Stage C configs | SuperPod Stage C evidence is recorded in `docs/fastwam_scheduler_sampler_plan.md`; scope is the recorded tasks, seeds, repeats, and runtime. Cross-backend scheduler adapters remain `planned`. |
| `dit_cache` | `native_cache` | FastWAM default `video_kv` path or `--opt dit_cache` where declared | FastWAM LIBERO/RoboTwin; DreamZero maps `dit_cache` to its server flag | `implemented` | Exact request-local cache hook with trace-visible status. It is not TeaCache, token pruning, or cross-replan cache. No catalog-level speedup claim without paired baseline evidence. |
| `cuda_graph` | `graph_compile` | FastWAM `cuda_graph(auto)` profile or runtime option | FastWAM action-body path with `dit_cache_mode=video_kv` | `measured` for documented FastWAM single-task runs | Maintainer SuperPod H800 evidence is summarized in `docs/optimization_integration.md`. The measured scope is the recorded FastWAM LIBERO/RoboTwin jobs; dynamic shapes or capture failure must fall back and stay trace-visible. |
| `torch_compile` | `graph_compile` | `--opt torch_compile` where declared | FastWAM action-body path | `experimental` | Existing evidence reports compile/fallback overhead; keep disabled by default until warmup and fallback behavior produce reportable gains. |
| `teacache` | `feature_cache` | `--opt teacache`, with `dit_cache_mode=video_kv` and TeaCache parameters | FastWAM action-only request-local path | `experimental` | `docs/fastwam_teacache_l1_report.md` records SuperPod runs. Observed call reduction and latency gains did not pass the output gate, so it is not `measured`. |
| `jpeg_observation_compression` | `output_control` | Cosmos-Policy profile/runtime context | Cosmos-Policy LIBERO | `implemented` | Exposed through the model entry and upstream runtime context; no measured speedup evidence yet. |
| `parallel_inference` | `batch_serving` | Cosmos-Policy profile/runtime context | Cosmos-Policy LIBERO | `experimental` | Multi-GPU/runtime path exists as an upstream toggle; EazyWAM has no measured product-path evidence yet. |
| `vla_cache` | `feature_cache` | none for current curated model entries | OpenVLA-family future backend target | `planned` / `unsupported` for current FastWAM, Cosmos-Policy, and DreamZero entries | Current manifests mark it unsupported for existing entries because it requires an OpenVLA model family. |

`fake_cache` is a fake backend test profile and is not a public acceleration
method.

## Unsupported Combinations

Make unsupported combinations explicit rather than silently omitting them:

- `vla_cache` is unsupported for current FastWAM, Cosmos-Policy, and DreamZero
  entries.
- Cosmos-Policy marks `dit_cache` unsupported because that flag is tied to
  DreamZero/FastWAM-style DiT cache hooks, not the Cosmos-Policy runtime path.
- TeaCache is unsupported on non-FastWAM entries and falls back when the
  FastWAM run does not use the exact `dit_cache(video_kv)` path.
- Batch or dynamic-shape serving may conflict with CUDA Graph capture until a
  backend declares stable batching/capture behavior.

## References

- `docs/acceleration_validation_runbook.md` - public validation loop and
  evidence templates.
- `docs/optimization_profiles.md` - profile-card fields and families.
- `docs/trace_schema.md` - profile status events and backend metadata.
- `docs/cli_entrypoints.md` - `--opt` on run, eval, and serve.
- `docs/contract.md` - run/eval/serve/compare product contracts.
- `docs/optimization_integration.md` - upstream integration notes and current
  FastWAM CUDA Graph evidence.
- `docs/fastwam_scheduler_sampler_plan.md` - FastWAM scheduler measured
  evidence.
- `docs/fastwam_teacache_l1_report.md` - FastWAM TeaCache L1 evidence and
  blockers.
