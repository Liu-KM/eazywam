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
| `scheduler` | `scheduler` | `--opt scheduler`, `--set num_inference_steps=...`, `--set sigma_shift=...` | FastWAM LIBERO and RoboTwin native eval paths | `measured` for documented FastWAM Stage C configs | Maintainer GPU Stage C evidence is recorded in `docs/fastwam_scheduler_sampler_plan.md`; scope is the recorded tasks, seeds, repeats, and runtime. Cross-backend scheduler adapters remain `planned`. |
| `dit_cache` | `native_cache` | FastWAM default `video_kv` path or `--opt dit_cache` where declared | FastWAM LIBERO/RoboTwin; DreamZero maps `dit_cache` to its server flag | `implemented` | Exact request-local cache hook with trace-visible status. It is not TeaCache, token pruning, or cross-replan cache. No catalog-level speedup claim without paired baseline evidence. |
| `cuda_graph` | `graph_compile` | FastWAM `cuda_graph(auto)` profile or runtime option | FastWAM action-body path with `dit_cache_mode=video_kv` | `measured` for documented FastWAM single-task runs | Maintainer H800 evidence is summarized in `docs/optimization_integration.md`. The measured scope is the recorded FastWAM LIBERO/RoboTwin runs; dynamic shapes or capture failure must fall back and stay trace-visible. |
| `torch_compile` | `graph_compile` | `--opt torch_compile` where declared | FastWAM action-body path | `experimental` | Existing evidence reports compile/fallback overhead; keep disabled by default until warmup and fallback behavior produce reportable gains. |
| `teacache` | `feature_cache` | `--opt teacache`, with `dit_cache_mode=video_kv` and TeaCache parameters | FastWAM action-only request-local path | `experimental` | `docs/fastwam_teacache_l1_report.md` records maintainer GPU runs. Observed call reduction and latency gains did not pass the compare/output gate, so it is not `measured`. |
| `jpeg_observation_compression` | `output_control` | Cosmos-Policy profile/runtime context | Cosmos-Policy LIBERO | `implemented` | Exposed through the model entry and upstream runtime context; no measured speedup evidence yet. |
| `parallel_inference` | `batch_serving` | Cosmos-Policy profile/runtime context | Cosmos-Policy LIBERO | `experimental` | Multi-GPU/runtime path exists as an upstream toggle; EazyWAM has no measured product-path evidence yet. |
| `vla_cache` | `feature_cache` | none for current curated model entries | OpenVLA-family future backend target | `planned` / `unsupported` for current FastWAM, Cosmos-Policy, and DreamZero entries | Current manifests mark it unsupported for existing entries because it requires an OpenVLA model family. |

`fake_cache` is a fake backend test profile and is not a public acceleration
method.

## Public-Safe Evidence Examples

These examples apply the Evidence Summary and Evidence Bundle v0.1 template
from `docs/acceleration_validation_runbook.md` to existing documented evidence.
They are public-safe summaries, not new benchmark runs. Raw private paths,
private scheduler wrappers, account names, partitions, login details, recovery
scripts, and large raw logs are intentionally excluded.

### `scheduler` Measured Scheduler-Profile Example

#### Evidence Summary

- Status decision: `measured` for the recorded FastWAM Stage C scheduler
  profile scope.
- Scope: `fastwam-libero` `libero-single-task` task `0` and
  `fastwam-robotwin` `robotwin-single-task` task `click_alarmclock`, seed `42`,
  three confirmation repeats per selected config, maintainer GPU runtime.
- Baseline: 10-step FastWAM FlowMatch Euler scheduler profile baseline.
- Variant: explicit `--opt scheduler` candidates,
  `num_inference_steps=2`, with `sigma_shift=3.25` for LIBERO and
  `sigma_shift=7` for RoboTwin.
- Task-quality result: success mean stayed `1.00` for both recorded targets.
- Performance result: total latency speedup mean was `1.761x` for LIBERO and
  `1.246x` for RoboTwin.
- Profile status and fallback: scheduler profile telemetry was trace-visible;
  the accepted recommended rows had no fallback.
- Output-drift audit: drift was recorded as an audit metric, not as a
  `parity_verified` claim; Stage C means were `0.0261` for LIBERO and
  `0.0123` for RoboTwin.
- Caveats and non-generalization notes: product defaults stay at the 10-step
  baseline; the measured claim is scoped to the documented tasks, seed, repeats,
  model entries, and runtime. Cross-backend scheduler adapters remain planned.

#### Evidence Bundle Example

```json
{
  "schema_version": "evidence_bundle.v0.1",
  "evidence_id": "fastwam-scheduler-stage-c-public-summary-2026-06-16",
  "summary": {
    "status_decision": "measured",
    "scope": "FastWAM LIBERO task 0 and RoboTwin click_alarmclock, seed 42, three confirmation repeats per selected config",
    "method_profile": "scheduler",
    "claim": "The selected scheduler-profile candidates reduced mean total latency while preserving the recorded success-rate gate for this scope.",
    "caveats": [
      "Measured status is scoped to the recorded Stage C configs and runtime.",
      "The evidence is not a parity_verified claim.",
      "The recommended configs are opt-in candidates, not new product defaults."
    ]
  },
  "method": {
    "name": "scheduler",
    "profile_family": "scheduler",
    "status_before": "experimental",
    "status_decision": "measured"
  },
  "scope": {
    "mode": "eval",
    "model_targets": [
      {
        "model_id": "fastwam-libero",
        "workload": "libero-single-task",
        "task_id": "0",
        "seed": 42,
        "repeat_count": 3,
        "trials_per_repeat": 1
      },
      {
        "model_id": "fastwam-robotwin",
        "workload": "robotwin-single-task",
        "task_name": "click_alarmclock",
        "task_config": "demo_randomized",
        "seed": 42,
        "repeat_count": 3,
        "episodes_per_repeat": 1
      }
    ]
  },
  "runtime_context": {
    "hardware_class": "maintainer GPU runtime",
    "accelerator_count": 1,
    "device": "cuda",
    "dtype": "bf16",
    "runtime_kind": "prepared FastWAM backend runtime",
    "eazywam_commit": "recorded in source evidence; omitted from public-safe example"
  },
  "commands": {
    "baseline_examples": [
      "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --opt scheduler --set seed=42 --set num_inference_steps=10 --set sigma_shift=null",
      "wam eval fastwam-robotwin --workload robotwin-single-task --task-name click_alarmclock --num-episodes 1 --set task_config=demo_randomized --opt scheduler --set seed=42 --set num_inference_steps=10 --set sigma_shift=null"
    ],
    "variant_examples": [
      "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --opt scheduler --set seed=42 --set num_inference_steps=2 --set sigma_shift=3.25",
      "wam eval fastwam-robotwin --workload robotwin-single-task --task-name click_alarmclock --num-episodes 1 --set task_config=demo_randomized --opt scheduler --set seed=42 --set num_inference_steps=2 --set sigma_shift=7"
    ],
    "compare": "python -m eazywam.evals.scheduler_acceptance <report>/scheduler_results.json --allow-missing-quality-reference --min-config-repeats 3"
  },
  "profile_status": {
    "requested": ["scheduler"],
    "applied": ["scheduler"],
    "fallback": false,
    "fallback_reason": null,
    "parameters": {
      "scheduler_name": "fastwam_flowmatch_euler",
      "baseline_num_inference_steps": 10,
      "libero_variant_num_inference_steps": 2,
      "libero_variant_sigma_shift": 3.25,
      "robotwin_variant_num_inference_steps": 2,
      "robotwin_variant_sigma_shift": 7
    }
  },
  "acceptance_validation": {
    "passed": true,
    "profile_trace_visible": true,
    "action_contract_passed": true,
    "task_quality_passed": true,
    "notes": "The accepted Stage C report includes scheduler metadata, trace paths, config summaries, repeat summaries, and conservative recommendation labels."
  },
  "measured_validation": {
    "passed": true,
    "primary_metric": "inference_end.timing.total_ms.mean",
    "results": [
      {
        "model_id": "fastwam-libero",
        "baseline_total_ms_mean": 157.50,
        "variant_total_ms_mean": 89.57,
        "speedup_mean": 1.761,
        "success_mean_baseline": 1.0,
        "success_mean_variant": 1.0
      },
      {
        "model_id": "fastwam-robotwin",
        "baseline_total_ms_mean": 313.17,
        "variant_total_ms_mean": 253.84,
        "speedup_mean": 1.246,
        "success_mean_baseline": 1.0,
        "success_mean_variant": 1.0
      }
    ]
  },
  "output_drift": {
    "available": true,
    "hard_gate": false,
    "results": [
      {"model_id": "fastwam-libero", "drift_mean": 0.0261},
      {"model_id": "fastwam-robotwin", "drift_mean": 0.0123}
    ],
    "summary": "Drift is an audit field for this evidence and does not establish parity_verified."
  },
  "artifacts": {
    "source_documents": [
      "docs/fastwam_scheduler_sampler_plan.md"
    ],
    "public_safe_trace_summary": "Stage C scheduler_results and scheduler_evidence_bundle summaries are described in the source document; raw local paths are not repeated here."
  },
  "privacy": {
    "public_safe": true,
    "redactions": [
      "raw private evidence roots",
      "private scheduler job ids",
      "private cache and upstream checkout paths"
    ],
    "excluded_private_details": [
      "scheduler submission files",
      "sbatch files",
      "account names",
      "partitions",
      "scratch paths",
      "login workflows",
      "credentials",
      "raw private logs",
      "site recovery scripts"
    ]
  }
}
```

### `cuda_graph` Measured Exact-Runtime Example

#### Evidence Summary

- Status decision: `measured` for the recorded FastWAM CUDA Graph action-body
  scope.
- Scope: `fastwam-libero` task `0` and `fastwam-robotwin`
  `click_alarmclock`, H800 maintainer runtime, `dit_cache_mode=video_kv`,
  `num_inference_steps=10`, single recorded task/episode scope.
- Baseline: eager cached FastWAM path with `cuda_graph_mode=off`.
- Variant: FastWAM `cuda_graph(auto)` action-body profile.
- Task-quality result: both baseline and variant runs succeeded in the recorded
  LIBERO and RoboTwin comparisons.
- Performance result: LIBERO mean total inference latency speedup was `1.91x`
  and mean denoise-loop speedup was `2.78x`; RoboTwin mean total latency
  speedup was `1.90x`.
- Profile status and fallback: `cuda_graph_capture_success=True`, median replay
  count `10`, and no capture fallback in the recorded evidence.
- Output-drift audit: no action-drift values are published in the catalog
  evidence; this example does not add an exact-output-preservation claim.
- Caveats and non-generalization notes: measured status is limited to the
  recorded stable-shape action-body path. Dynamic shapes, capture failure, or
  serving behavior must be validated separately.

#### Evidence Bundle Example

```json
{
  "schema_version": "evidence_bundle.v0.1",
  "evidence_id": "fastwam-cuda-graph-public-summary-2026-06-30",
  "summary": {
    "status_decision": "measured",
    "scope": "FastWAM LIBERO task 0 and RoboTwin click_alarmclock on H800 maintainer runtime, cuda_graph(auto), dit_cache_mode=video_kv",
    "method_profile": "cuda_graph",
    "claim": "CUDA Graph reduced recorded FastWAM action-body latency with successful task runs and no capture fallback for this scope.",
    "caveats": [
      "Measured status applies only to the recorded stable-shape FastWAM action-body path.",
      "Dynamic-shape, batch, and resident serving behavior are not validated by this evidence.",
      "This example does not add an exact-output-preservation claim."
    ]
  },
  "method": {
    "name": "cuda_graph",
    "profile_family": "graph_compile",
    "status_before": "implemented",
    "status_decision": "measured"
  },
  "scope": {
    "mode": "eval",
    "model_targets": [
      {
        "model_id": "fastwam-libero",
        "workload": "libero-single-task",
        "task_id": "0",
        "seed": 42,
        "trial_count": 1,
        "num_inference_steps": 10,
        "action_horizon": 32,
        "replan_steps": 10
      },
      {
        "model_id": "fastwam-robotwin",
        "workload": "robotwin-single-task",
        "task_name": "click_alarmclock",
        "task_config": "demo_randomized",
        "episode_count": 1
      }
    ]
  },
  "runtime_context": {
    "hardware_class": "H800",
    "accelerator_count": 1,
    "device": "cuda",
    "dtype": "bf16",
    "runtime_kind": "prepared FastWAM backend runtime",
    "eazywam_commit": "recorded in source evidence; omitted from public-safe example"
  },
  "commands": {
    "baseline_examples": [
      "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --set seed=42 --set num_inference_steps=10 --set action_horizon=32 --set replan_steps=10 --set dit_cache_mode=video_kv --set cuda_graph_mode=off",
      "wam eval fastwam-robotwin --workload robotwin-single-task --task-name click_alarmclock --num-episodes 1 --set task_config=demo_randomized --set dit_cache_mode=video_kv --set cuda_graph_mode=off"
    ],
    "variant_examples": [
      "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --set seed=42 --set num_inference_steps=10 --set action_horizon=32 --set replan_steps=10 --set dit_cache_mode=video_kv",
      "wam eval fastwam-robotwin --workload robotwin-single-task --task-name click_alarmclock --num-episodes 1 --set task_config=demo_randomized --set dit_cache_mode=video_kv"
    ],
    "compare": "wam compare <baseline-trace> <cuda-graph-trace>"
  },
  "profile_status": {
    "requested": ["cuda_graph"],
    "applied": ["cuda_graph"],
    "fallback": false,
    "fallback_reason": null,
    "parameters": {
      "mode": "auto",
      "capture": "action_body",
      "requires": "dit_cache_mode=video_kv"
    },
    "telemetry": {
      "cuda_graph_capture_success": true,
      "median_replay_count": 10
    }
  },
  "acceptance_validation": {
    "passed": true,
    "profile_trace_visible": true,
    "action_contract_passed": true,
    "task_quality_passed": true,
    "notes": "Recorded evidence reports successful runs, capture success, replay telemetry, and no fallback."
  },
  "measured_validation": {
    "passed": true,
    "primary_metric": "inference_end.timing.total_ms.mean",
    "results": [
      {
        "model_id": "fastwam-libero",
        "total_ms_speedup": 1.91,
        "denoise_wall_ms_speedup": 2.78,
        "eval_duration_speedup": 1.39,
        "success_rate_baseline": 1.0,
        "success_rate_variant": 1.0
      },
      {
        "model_id": "fastwam-robotwin",
        "baseline_total_ms_mean": 651.08,
        "variant_total_ms_mean": 342.85,
        "total_ms_speedup": 1.90,
        "success_rate_baseline": 1.0,
        "success_rate_variant": 1.0
      }
    ]
  },
  "output_drift": {
    "available": false,
    "hard_gate": false,
    "summary": "The existing public evidence records task success and runtime telemetry. No output-drift values are published in the catalog evidence, and this example does not claim exact output preservation."
  },
  "artifacts": {
    "source_documents": [
      "docs/optimization_integration.md",
      "docs/backends.md"
    ],
    "public_safe_trace_summary": "The source documents summarize the recorded latency, success, capture, replay, and fallback results without requiring private scheduler operations."
  },
  "privacy": {
    "public_safe": true,
    "redactions": [
      "raw private trace roots",
      "private scheduler job ids",
      "private cache and upstream checkout paths"
    ],
    "excluded_private_details": [
      "scheduler submission files",
      "sbatch files",
      "account names",
      "partitions",
      "scratch paths",
      "login workflows",
      "credentials",
      "raw private logs",
      "site recovery scripts"
    ]
  }
}
```

### `teacache` Experimental Evidence Example

#### Evidence Summary

- Status decision: keep `experimental`; do not promote TeaCache L1 to
  `measured`.
- Scope: FastWAM action-only request-local TeaCache L1 on
  `dit_cache_mode=video_kv` with `cuda_graph_mode=off`, LIBERO task `0` and
  RoboTwin `click_alarmclock`, seed `42`, five trials or episodes per target.
- Baseline: eager cached FastWAM path with TeaCache disabled.
- Variant: `--opt teacache`, `teacache_threshold=0.05`,
  `teacache_warmup_steps=1`.
- Task-quality result: both recorded targets kept success rate `1.0`.
- Performance result: observed speedup-like signals were recorded for
  auditability only: LIBERO total latency `1.547x`, RoboTwin total latency
  `1.427x`, with non-zero hit rates and skipped steps.
- Profile status and fallback: TeaCache telemetry was recorded, but the report
  table does not publish final fallback values.
- Output-drift audit: compare reported `compare_decision=invalid` and
  `output_gate_not_passed` for both targets. Drift is treated as an audit and
  measured-status blocker for this evidence, not as an exact-output claim.
- Caveats and non-generalization notes: the evidence shows implementation and
  speedup-like signals, but not a reportable speedup. TeaCache remains disabled
  by default and separate from exact `dit_cache(video_kv)`.

#### Evidence Bundle Example

```json
{
  "schema_version": "evidence_bundle.v0.1",
  "evidence_id": "fastwam-teacache-l1-public-summary-2026-06-30",
  "summary": {
    "status_decision": "experimental",
    "scope": "FastWAM TeaCache L1 on LIBERO task 0 and RoboTwin click_alarmclock, dit_cache_mode=video_kv, cuda_graph_mode=off",
    "method_profile": "teacache",
    "claim": "TeaCache L1 is selectable and produced trace-visible cache activity, but the existing comparison did not justify measured status.",
    "caveats": [
      "Observed speedup-like values are audit data, not reportable speedup claims.",
      "Both recorded targets have compare_decision=invalid and output_gate_not_passed blockers.",
      "TeaCache L1 is approximate, disabled by default, and not native/reference parity evidence."
    ]
  },
  "method": {
    "name": "teacache",
    "profile_family": "feature_cache",
    "status_before": "implemented",
    "status_decision": "experimental"
  },
  "scope": {
    "mode": "eval",
    "model_targets": [
      {
        "model_id": "fastwam-libero",
        "workload": "libero-single-task",
        "task_id": "0",
        "seed": 42,
        "trial_count": 5,
        "num_inference_steps": 20
      },
      {
        "model_id": "fastwam-robotwin",
        "workload": "robotwin-single-task",
        "task_name": "click_alarmclock",
        "task_config": "demo_randomized",
        "seed": 42,
        "episode_count": 5,
        "num_inference_steps": 20
      }
    ]
  },
  "runtime_context": {
    "hardware_class": "maintainer GPU runtime",
    "accelerator_count": 1,
    "device": "cuda",
    "dtype": "bf16",
    "runtime_kind": "prepared FastWAM backend runtime",
    "eazywam_commit": "recorded in source evidence; omitted from public-safe example"
  },
  "commands": {
    "baseline_examples": [
      "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 5 --set seed=42 --set num_inference_steps=20 --set dit_cache_mode=video_kv --set cuda_graph_mode=off",
      "wam eval fastwam-robotwin --workload robotwin-single-task --task-name click_alarmclock --num-episodes 5 --set task_config=demo_randomized --set seed=42 --set num_inference_steps=20 --set dit_cache_mode=video_kv --set cuda_graph_mode=off"
    ],
    "variant_examples": [
      "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 5 --opt teacache --set seed=42 --set num_inference_steps=20 --set dit_cache_mode=video_kv --set cuda_graph_mode=off --set teacache_threshold=0.05 --set teacache_warmup_steps=1",
      "wam eval fastwam-robotwin --workload robotwin-single-task --task-name click_alarmclock --num-episodes 5 --set task_config=demo_randomized --opt teacache --set seed=42 --set num_inference_steps=20 --set dit_cache_mode=video_kv --set cuda_graph_mode=off --set teacache_threshold=0.05 --set teacache_warmup_steps=1"
    ],
    "compare_examples": [
      "wam compare <fastwam-libero-baseline-trace> <fastwam-libero-teacache-trace> --max-action-drift 0.001",
      "wam compare <fastwam-robotwin-baseline-trace> <fastwam-robotwin-teacache-trace> --max-action-drift 0.001"
    ]
  },
  "profile_status": {
    "requested": ["teacache"],
    "applied": ["teacache"],
    "fallback": null,
    "fallback_reason": "not published in the report table",
    "parameters": {
      "dit_cache_mode": "video_kv",
      "cuda_graph_mode": "off",
      "teacache_threshold": 0.05,
      "teacache_warmup_steps": 1
    }
  },
  "acceptance_validation": {
    "passed": true,
    "profile_trace_visible": true,
    "action_contract_passed": true,
    "task_quality_passed": true,
    "notes": "Recorded runs show TeaCache profile selection, non-zero hit/skipped-step telemetry, and unchanged success rate."
  },
  "measured_validation": {
    "passed": false,
    "primary_metric": "latency_ms.mean",
    "blockers": [
      "compare_decision:invalid",
      "output_gate_not_passed"
    ],
    "observed_audit_values": [
      {
        "model_id": "fastwam-libero",
        "latency_mean_speedup_observed": 1.547,
        "denoise_mean_speedup_observed": 1.653,
        "teacache_hit_rate": 0.3974,
        "teacache_skipped_steps": 7.949,
        "teacache_drift_score": 0.3665,
        "success_rate_baseline": 1.0,
        "success_rate_variant": 1.0,
        "speedup_reportable": false
      },
      {
        "model_id": "fastwam-robotwin",
        "latency_mean_speedup_observed": 1.427,
        "denoise_mean_speedup_observed": 1.525,
        "teacache_hit_rate": 0.35,
        "teacache_skipped_steps": 7.0,
        "teacache_drift_score": 0.3137,
        "success_rate_baseline": 1.0,
        "success_rate_variant": 1.0,
        "speedup_reportable": false
      }
    ]
  },
  "output_drift": {
    "available": true,
    "hard_gate": false,
    "passed": false,
    "results": [
      {"model_id": "fastwam-libero", "action_drift": 0.9731},
      {"model_id": "fastwam-robotwin", "action_drift": 0.03143}
    ],
    "summary": "The existing report used action drift in its compare output and blocked measured status. This is not an exact-output-preservation claim."
  },
  "artifacts": {
    "source_documents": [
      "docs/fastwam_teacache_l1_report.md"
    ],
    "public_safe_trace_summary": "The source report records summary rows and blockers; raw local trace roots are not repeated here."
  },
  "privacy": {
    "public_safe": true,
    "redactions": [
      "raw private evidence roots",
      "private scheduler job ids",
      "private cache and upstream checkout paths"
    ],
    "excluded_private_details": [
      "scheduler submission files",
      "sbatch files",
      "account names",
      "partitions",
      "scratch paths",
      "login workflows",
      "credentials",
      "raw private logs",
      "site recovery scripts"
    ]
  }
}
```

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
