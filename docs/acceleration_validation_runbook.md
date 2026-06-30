# Acceleration Validation Runbook

This runbook defines the public EazyWAM validation loop for acceleration
methods. It is written for any prepared GPU runtime that can run the standard
`wam` command. It is not a SuperPod runbook, a cluster scheduler guide, or a
private maintainer operations guide.

The first validation path covers:

- `wam run` for explicit-observation acceptance checks.
- `wam eval` for workload and task-quality checks.
- `wam compare` for baseline-to-variant trace comparison.

`wam serve` acceleration validation is a required follow-up track, described
below, but it is not part of this first main path.

## Public Boundary

Public validation evidence should describe the EazyWAM product path:

- model id, workload, mode, seed, task id, and trial count.
- baseline and variant `wam` commands.
- optimization profile requested, applied, fallback state, and parameters.
- trace path, compare summary, eval summary, or redacted trace summary.
- hardware class, accelerator count, device, dtype, backend runtime kind, and
  EazyWAM commit.
- task-quality result, latency or memory result, output-drift audit result when
  comparable, and caveats.

Public docs must not include private maintainer runtime instructions. Do not
publish SuperPod operational details, scheduler submission files, sbatch files,
account names, partitions, scratch paths, recovery scripts, login workflows,
credentials, or raw private logs. If a maintainer runtime produced evidence,
record only
public-safe runtime context such as hardware class, accelerator count, dtype,
runtime kind, and the standard `wam` commands that were run.

## Validation Levels

### Acceptance Validation

Acceptance Validation answers: "Is this method usable through EazyWAM for this
declared scope?"

It proves that an acceleration method is:

- selectable through `--opt <method>` or an equivalent declared profile path.
- visible in traces through `optimization_profile_status`, runtime info, or
  backend metadata.
- compatible with the EazyWAM action/result contract for at least one relevant
  small run or eval.
- able to complete without obvious task-quality failure, invalid action shape,
  hidden fallback, or missing trace metadata.

Acceptance Validation can use the same artifacts that later support Measured
Validation, but it does not prove a reportable speedup. Passing acceptance
usually supports `experimental` status, not `measured`.

### Measured Validation

Measured Validation answers: "Is this a scoped, reportable speed or memory
improvement?"

It requires Acceptance Validation plus a baseline and variant comparison under a
declared scope:

- model id and workload.
- baseline command with the method disabled or not requested.
- variant command with `--opt <method>` or the equivalent profile declaration.
- runtime context, including hardware class, accelerator count, device, dtype,
  backend runtime kind, and EazyWAM commit.
- seed, task scope, trial count, and any workload overrides.
- fallback policy and observed requested/applied/fallback state.
- task-quality gate, such as simulator success rate or episode metrics when
  relevant.
- output-drift audit when comparable. Drift is a hard gate only for methods
  that explicitly claim exact output preservation.
- latency or memory comparison from `wam compare`, eval summaries, or a
  public-safe trace summary.

Only Measured Validation can promote a method to scoped `measured` status.

## Standard Loop

### 1. Declare The Scope

Write the scope before running commands:

- method/profile: for example `scheduler`, `cuda_graph`, or `teacache`.
- model id: for example `fastwam-libero`.
- mode: `run` or `eval`.
- workload and task scope when using eval.
- seed and trial count when the workload supports them.
- runtime context: generic GPU class, accelerator count, dtype, backend runtime
  kind, and EazyWAM commit.
- expected status decision if the run passes.

### 2. Run A Baseline

For explicit-observation acceptance or latency smoke:

```bash
wam run fastwam-libero \
  --input obs.json \
  --output runs/validation/baseline/action.json \
  --cache-dir /mnt/wam-cache \
  --trace-dir runs/validation/baseline
```

For workload validation:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /mnt/wam-cache \
  --trace-dir runs/validation/baseline \
  --summary-path runs/validation/baseline/eval-summary.json
```

Use the same prepared runtime, cache boundary, model assets, workload, task
scope, seed, and trial count for the variant unless the evidence explicitly
states why a field differs.

### 3. Run The Variant

For explicit-observation acceptance:

```bash
wam run fastwam-libero \
  --input obs.json \
  --output runs/validation/variant/action.json \
  --cache-dir /mnt/wam-cache \
  --trace-dir runs/validation/variant \
  --opt scheduler
```

For workload validation with profile parameters:

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /mnt/wam-cache \
  --trace-dir runs/validation/variant \
  --summary-path runs/validation/variant/eval-summary.json \
  --opt scheduler \
  --set num_inference_steps=6 \
  --set sigma_shift=3.0
```

For `wam run`, profile-specific request parameters should come from the input
file's `runtime_options` when the method needs request-local values. Do not add
method-specific public flags for one backend.

### 4. Compare The Runs

Compare trace files or directories:

```bash
wam compare runs/validation/baseline runs/validation/variant
```

Use `--max-action-drift` only when the method and workload have a declared
numeric drift tolerance:

```bash
wam compare \
  runs/validation/baseline/trace.jsonl \
  runs/validation/variant/trace.jsonl \
  --max-action-drift 0.001
```

Record the decision, primary metric, output gate, runtime contract gate, metric
comparisons, profile telemetry, warnings, and errors.

### 5. Decide The Status

Use the catalog status labels without changing their meanings:

| Status | Validation decision |
|---|---|
| `planned` | The method is in scope, but there is no stable EazyWAM enablement path yet. Evidence may describe intended scope, but no acceptance result exists. |
| `implemented` | Code and profile enablement exist, but Acceptance Validation has not passed or has not been recorded. Do not claim speedup. |
| `experimental` | Acceptance Validation passed, or the method is testable, but Measured Validation is missing, blocked, unstable, fallback-prone, or fails quality/output gates. Do not claim proven speedup. |
| `measured` | Measured Validation passed for the declared model, workload, runtime, fallback policy, quality gate, and performance metric. The claim is scoped to that evidence. |
| `unsupported` | The method is known not to apply to the model, backend, workload, device, runtime path, or profile combination. Record the reason instead of silently omitting it. |

If a run shows speedup-like latency but fails task quality, action shape,
runtime contract, or required exact-output gates, keep the method
`experimental` or `unsupported` for that scope.

## Human Evidence Summary Template

Use this short form first so reviewers can understand the claim before reading
the JSON bundle.

```markdown
## Evidence Summary

- Status decision:
- Scope:
- Method/profile:
- Model and workload:
- Baseline command:
- Variant command:
- Task-quality result:
- Performance result:
- Profile status and fallback:
- Output-drift audit:
- Artifacts:
- Caveats and non-generalization notes:
- Public/private boundary check:
```

## Evidence Bundle v0.1 Example

This JSON shape is an example for agents and future tooling. It is not a formal
schema package yet.

```json
{
  "schema_version": "evidence_bundle.v0.1",
  "evidence_id": "fastwam-libero-scheduler-libero-single-task-2026-06-30",
  "summary": {
    "status_decision": "measured",
    "scope": "fastwam-libero, libero-single-task task 0, one prepared GPU runtime, bf16, one trial",
    "method_profile": "scheduler",
    "claim": "Variant reduced mean inference latency while preserving the declared task-quality gate for this scope.",
    "caveats": [
      "Measured status applies only to the recorded model, workload, runtime, seed, task id, and trial count.",
      "This evidence does not validate resident serving behavior."
    ]
  },
  "method": {
    "name": "scheduler",
    "profile_family": "scheduler",
    "status_before": "implemented",
    "status_decision": "measured"
  },
  "scope": {
    "model_id": "fastwam-libero",
    "mode": "eval",
    "workload": "libero-single-task",
    "task_id": "0",
    "seed": null,
    "trial_count": 1,
    "task_scope": "single-task smoke"
  },
  "runtime_context": {
    "hardware_class": "single CUDA GPU",
    "accelerator_count": 1,
    "device": "cuda:0",
    "dtype": "bf16",
    "runtime_kind": "prepared backend runtime",
    "eazywam_commit": "REPLACE_WITH_COMMIT"
  },
  "commands": {
    "baseline": "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --cache-dir /mnt/wam-cache --trace-dir runs/validation/baseline --summary-path runs/validation/baseline/eval-summary.json",
    "variant": "wam eval fastwam-libero --workload libero-single-task --task-id 0 --num-trials 1 --cache-dir /mnt/wam-cache --trace-dir runs/validation/variant --summary-path runs/validation/variant/eval-summary.json --opt scheduler --set num_inference_steps=6 --set sigma_shift=3.0",
    "compare": "wam compare runs/validation/baseline runs/validation/variant"
  },
  "profile_status": {
    "requested": ["scheduler"],
    "applied": ["scheduler"],
    "fallback": false,
    "fallback_reason": null,
    "parameters": {
      "num_inference_steps": 6,
      "sigma_shift": 3.0
    }
  },
  "acceptance_validation": {
    "passed": true,
    "profile_trace_visible": true,
    "action_contract_passed": true,
    "task_quality_passed": true,
    "notes": "Variant completed the declared workload and emitted trace-visible scheduler metadata."
  },
  "measured_validation": {
    "passed": true,
    "primary_metric": "latency_ms.mean",
    "baseline_value": 1000.0,
    "variant_value": 760.0,
    "relative_change": -0.24,
    "speedup": 1.32,
    "memory_result": null,
    "quality_gate": {
      "metric": "success_rate",
      "baseline": 1.0,
      "variant": 1.0,
      "passed": true
    }
  },
  "output_drift": {
    "available": true,
    "hard_gate": false,
    "max_action_drift": 0.001,
    "passed": true,
    "summary": "Action summary drift stayed within the declared audit tolerance."
  },
  "artifacts": {
    "baseline_trace": "runs/validation/baseline/trace.jsonl",
    "variant_trace": "runs/validation/variant/trace.jsonl",
    "baseline_summary": "runs/validation/baseline/eval-summary.json",
    "variant_summary": "runs/validation/variant/eval-summary.json",
    "compare_summary": "captured JSON stdout from wam compare"
  },
  "privacy": {
    "public_safe": true,
    "redactions": [],
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

When Measured Validation fails, keep the same bundle shape but set
`measured_validation.passed` to `false`, explain the blocker, and choose
`experimental`, `implemented`, or `unsupported` as the status decision.

## Updating The Acceleration Catalog

Every accepted evidence record should update
`docs/acceleration_methods.md` when it changes public user expectations.

For a catalog update:

- keep the method name, profile family, enablement path, applicability, status,
  and evidence notes in one row.
- keep status scoped. Write `measured for <model/workload/runtime scope>` when
  the evidence is scoped.
- link to a public-safe evidence summary, report, or runbook section instead of
  raw private logs.
- record unsupported combinations explicitly when validation proves a method
  cannot apply.
- do not promote `implemented` or `experimental` methods as proven speedups.
- do not paste private runtime operations into the catalog.

If the evidence would require changing status meanings, public command behavior,
or the SuperPod public/private boundary, record the concern and do not update
the method to `measured` until the product decision is made.

## `wam serve` Follow-Up Track

Serving validation is required, but it has a different risk shape than the first
`run`/`eval`/`compare` path. A resident server has startup, warmup, health,
steady-state request latency, request failures, batching, shape changes,
long-running stability, and cleanup behavior that a single `wam run` or
simulator `wam eval` does not cover.

The follow-up serving track should define:

- startup and warmup evidence from `serve_start`, `serve_ready`, and
  `backend_close`.
- request-latency evidence from repeated `/infer` calls and
  `serve_request_end`.
- health-check and error behavior.
- long-running stability and memory behavior.
- batching and dynamic-shape behavior when a backend declares support.
- profile fallback rules for server-scoped methods such as batch serving or
  CUDA Graph under changing request shapes.

Until that track exists, do not use `wam serve` evidence alone to promote a
method to `measured` for non-serving paths, and do not use `run` or `eval`
evidence alone to claim resident serving acceleration.
