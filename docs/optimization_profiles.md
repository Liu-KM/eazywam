# Optimization Profiles

EazyWAM treats inference acceleration methods as explicit optimization
profiles. A profile is a contract: it defines the user-facing name, runtime
scope, parameters, trace fields, output checks, and rollout status for one
optimization family.

This page is a contract document. It does not imply that every listed profile
has a working implementation. Model manifests and runtime traces decide what is
supported, enabled, applied, measured, or still experimental.

See `docs/acceleration_methods.md` for the public acceleration method catalog,
method status labels, and minimum measured evidence standard. A method is the
backend-integrated implementation; a profile is the EazyWAM control and
validation contract for enabling and comparing that method.

The current implementation focus is single-request FastWAM inference
acceleration: `dit_cache(video_kv)`, `cuda_graph(auto)`, the opt-in `scheduler`
profile, TeaCache L1 as an opt-in approximate cache, and experimental opt-in
`torch_compile`. Deferred batch serving is outside the current product path;
this document does not define a batch-inference API or a new batch runner
commitment.

Rollout evidence is tracked separately from implementation status. Before a
FastWAM scheduler, TeaCache, or future approximate acceleration profile moves
from opt-in experimental use to `recommended_candidate`, `parity_verified`, or
`default_enabled`, it must satisfy the full-task or representative-task gates in
`docs/fastwam_full_task_evidence_plan.md`. Existing single-task scheduler and
TeaCache evidence must not be used as a default-enable claim.

## Profile Card Shape

Every profile should be documented with this shape before implementation:

```yaml
name:
family:
class:
default:
scope:
parameters:
requires:
conflicts:
trace_fields:
output_check:
status:
```

Field meanings:

| Field | Meaning |
|---|---|
| `name` | Stable CLI and manifest name. |
| `family` | One of the profile families below. |
| `class` | `training_free_inference`, `runtime_system`, `post_training`, `training_recipe`, or `model_architecture`. |
| `default` | `enabled`, `disabled`, or `backend_default`. |
| `scope` | `request`, `replan`, `episode`, `run`, `server`, `simulator_eval`, or `native_backend`. |
| `parameters` | Typed user-facing parameters and defaults. |
| `requires` | Backend capabilities, device requirements, shape constraints, or runtime dependencies. |
| `conflicts` | Profiles or runtime modes that cannot be enabled together. |
| `trace_fields` | Required trace or backend metadata fields. |
| `output_check` | How behavior is compared with the profile disabled. |
| `status` | `planned`, `implemented`, `experimental`, `measured`, `unsupported`, or `deferred` for intentionally parked profile records; method status is scoped by model/workload/runtime evidence. |

## Profile Families

| Family | Purpose | Examples |
|---|---|---|
| `action_runtime` | Reduce model calls by controlling action chunks and replanning. | `action_chunk_scheduling`, future `action_runtime`. |
| `output_control` | Avoid unnecessary WAM outputs and artifacts. | `action_only`, no future video decode/save. |
| `scheduler` | Control diffusion or flow sampling schedules. | DPM-Solver++, UniPC, AYS, Karras, FlowMatch schedules. |
| `dtype` | Control model/runtime precision and matmul policy. | bf16, fp16, fp32, TF32. |
| `attention_backend` | Select transformer attention implementation. | SDPA, FlashAttention, xFormers, SageAttention. |
| `native_cache` | Use model-native request-local caches. | FastWAM `dit_cache` / `video_kv`. |
| `feature_cache` | Reuse or approximate intermediate DiT features. | `teacache`, PAB, FasterCache. |
| `graph_compile` | Reduce Python/kernel launch overhead with graph or compile paths. | `cuda_graph`, `torch_compile`. |
| `batch_serving` | Deferred throughput/serving family; not in the current product path. | Deferred eval sharding, dynamic batch serving, batched action denoise. |

## Standard Profile Specs

### `action_chunk_scheduling`

```yaml
name: action_chunk_scheduling
family: action_runtime
class: training_free_inference
default: backend_default
scope: simulator_eval
parameters:
  action_horizon: manifest.defaults.action_horizon
  replan_steps: manifest.defaults.replan_steps
  execute_horizon: null
  temporal_ensemble: false
requires:
  backend_capabilities: [action_chunks]
conflicts: []
trace_fields:
  - action_horizon
  - replan_steps
  - execute_horizon
  - chunk_size
  - tail_actions_dropped
  - from_stale_chunk
output_check: action_shape_and_success_rate
status: implemented
```

`action_chunk_scheduling` is the current compatibility profile for action
chunking and receding horizon control. Future work may add `execute_horizon`
and `temporal_ensemble`, but the current runner already consumes action chunks
and replans through `action_horizon` and `replan_steps`.

### `action_runtime`

```yaml
name: action_runtime
family: action_runtime
class: training_free_inference
default: disabled
scope: episode
parameters:
  chunk_size: null
  execute_horizon: null
  temporal_ensemble: false
  temporal_ensemble_window: null
requires:
  backend_capabilities: [action_chunks]
conflicts: []
trace_fields:
  - action_horizon
  - execute_horizon
  - replan_steps
  - temporal_ensemble_enabled
  - temporal_ensemble_window
  - tail_actions_dropped
output_check: action_drift_or_success_rate
status: planned
```

This is the future umbrella profile for policy-runtime controls. It should not
replace `action_chunk_scheduling` until the CLI and manifests have a migration
path.

### `output_control`

```yaml
name: output_control
family: output_control
class: training_free_inference
default: backend_default
scope: request
parameters:
  action_only: true
  return_future: false
  video_decode: false
  video_save: false
  keep_future_latent: false
requires:
  backend_capabilities: [action_output]
conflicts: []
trace_fields:
  - action_only
  - return_future
  - video_decode_enabled
  - video_save_enabled
  - future_latent_kept
output_check: action_shape_and_optional_future_artifacts
status: implemented
```

FastWAM already uses an action-only native inference path and its manifests
default `return_future: false`. The cross-model profile contract still needs a
uniform request/runtime option.

### `scheduler`

```yaml
name: scheduler
family: scheduler
class: training_free_inference
default: backend_default
scope: request
parameters:
  num_inference_steps: null
  sigma_shift: null
  scheduler_name: backend_default
  solver: backend_default
  schedule_type: backend_default
  timesteps: null
  sigmas: null
requires:
  backend_capabilities: [diffusion_or_flow_sampler]
conflicts: []
trace_fields:
  - scheduler_name
  - solver
  - num_inference_steps
  - sigma_shift
  - timestep_count
  - timesteps
  - sigmas
  - schedule_type
  - schedule_source
  - denoise_wall_ms
  - total_ms
output_check: action_drift_or_success_rate
status: measured
```

Target solvers and schedules include DPM-Solver++, UniPC, AYS, Karras, and
FlowMatch Euler/Heun adapters. This contract should call backend or Diffusers
schedulers where possible instead of reimplementing solvers in the harness
core.

FastWAM currently exposes a training-free `scheduler` profile for its native
shifted FlowMatch Euler action scheduler. The default profile parameters are
null for `num_inference_steps`, `sigma_shift`, `timesteps`, and `sigmas`, so
enabling the profile without overrides does not change the current 10-step
FastWAM LIBERO/RoboTwin eval baseline. Users can sweep values through eval
manifest overrides, for example `wam eval fastwam-libero --opt scheduler --set
num_inference_steps=6 --set sigma_shift=3.0`, or through request
`runtime_options` on resident native inference paths. `timesteps` and `sigmas`
are mutually exclusive Diffusers-style custom schedule inputs; when provided,
FastWAM traces `schedule_source` as `custom_timesteps` or `custom_sigmas`.

### `dtype`

```yaml
name: dtype
family: dtype
class: runtime_system
default: backend_default
scope: run
parameters:
  runtime_dtype: backend_default
  weight_dtype: backend_default
  activation_dtype: backend_default
  tf32: backend_default
requires:
  device: cuda_or_cpu
conflicts: []
trace_fields:
  - runtime_dtype
  - weight_dtype
  - activation_dtype
  - tf32_enabled
output_check: action_shape_and_success_rate
status: implemented
```

Manifests already record dtype defaults such as `bf16`. The unified profile
should make dtype and TF32 behavior trace-visible across backends.

### `attention_backend`

```yaml
name: attention_backend
family: attention_backend
class: runtime_system
default: backend_default
scope: run
parameters:
  backend: sdpa
  allow_fallback: true
requires:
  backend_capabilities: [transformer_attention]
conflicts: []
trace_fields:
  - attention_backend
  - attention_backend_actual
  - attention_backend_fallback_reason
output_check: action_drift_or_success_rate
status: planned
```

The first supported values should be `sdpa`, `flash_attn`, and `xformers`.
`sageattention` belongs behind the same interface but should remain optional
and experimental until success-rate evidence exists.

### `dit_cache`

```yaml
name: dit_cache
family: native_cache
class: training_free_inference
default: backend_default
scope: native_backend
parameters:
  mode: video_kv
requires:
  backend_capabilities: [fastwam_video_kv_cache]
conflicts: []
trace_fields:
  - dit_cache_enabled
  - dit_cache_mode
  - dit_cache_hook
  - video_seq_len
  - action_seq_len
  - cache_layers
  - cache_prefill_wall_ms
  - cache_bytes
output_check: action_shape_and_success_rate
status: implemented
```

`dit_cache` currently means the FastWAM request-local video K/V cache. It is not
TeaCache, not cross-replan cache, not token pruning, and not step skipping.

### `teacache`

```yaml
name: teacache
family: feature_cache
class: training_free_inference
default: disabled
scope: request
parameters:
  mode: auto
  threshold: null
  layers: null
  warmup_steps: null
requires:
  backend_capabilities: [dit_feature_cache_hooks]
conflicts:
  - incompatible_scheduler_modes
trace_fields:
  - teacache_enabled
  - teacache_mode
  - teacache_layers
  - teacache_threshold
  - teacache_warmup_steps
  - teacache_hit_rate
  - teacache_skipped_steps
  - teacache_drift_score
  - teacache_fallback_reason
output_check: action_drift_and_success_rate
status: experimental
```

TeaCache is the FastWAM approximate feature-cache target. L1 is opt-in,
request-local, action-only, and only applies on the FastWAM `dit_cache`
`video_kv` path. It is implemented as a step-output cache around action denoise
steps, not as layer-level TeaCache. It must remain separate from `dit_cache`,
and measurements must report action drift and simulator success rate rather than
claiming native/reference parity. The current TeaCache L1 product path is
single-request only. Deferred batch serving for TeaCache stays with the rest of
the throughput/serving work; current profile docs should not imply an active
batch-inference path or batch-specific fallback contract.

### `cuda_graph`

```yaml
name: cuda_graph
family: graph_compile
class: runtime_system
default: backend_default
scope: native_backend
parameters:
  mode: auto
  capture: action_body
requires:
  device: cuda
  backend_capabilities: [static_action_body_capture]
  profiles: [dit_cache]
conflicts:
  - deferred_batch_serving
trace_fields:
  - cuda_graph_enabled
  - cuda_graph_mode
  - cuda_graph_capture_success
  - cuda_graph_replay_count
  - cuda_graph_fallback_reason
  - cuda_graph_shape_key
output_check: action_shape_and_success_rate
status: measured
```

FastWAM currently captures only the action-body
`mot.forward_action_with_video_cache()` path after the `video_kv` cache has
been prefetched.

### `torch_compile`

```yaml
name: torch_compile
family: graph_compile
class: runtime_system
default: disabled
scope: native_backend
parameters:
  mode: auto
  target: action_body
requires:
  backend_capabilities: [compile_action_body]
conflicts: []
trace_fields:
  - torch_compile_enabled
  - torch_compile_mode
  - torch_compile_success
  - torch_compile_fallback_reason
  - torch_compile_wall_ms
output_check: latency_after_warmup_and_success_rate
status: experimental
```

`torch_compile` stays opt-in. FastWAM evidence shows that compile overhead and
Inductor fallback can make first-request latency worse than CUDA Graph alone.

### `batch_serving` (deferred record)

This deferred record preserves the old throughput idea for future design only.
It is a deferred note, not a current implementation spec, and does not define
a batch-inference API or a batch runner.

```yaml
name: batch_serving
family: batch_serving
class: runtime_system
default: disabled
scope: server
parameters:
  design_status: deferred
  implementation_commitment: none
requires:
  backend_capabilities: []
conflicts:
  - current_single_request_mainline
trace_fields: []
output_check: fresh_design_required
status: deferred
```

Deferred status: batch serving is deferred. The first prototype was removed from
the product path because it mixed serving, eval orchestration, and backend
batching too early. Revisit this profile after the single-request acceleration
path is stable, with a cleaner serving contract and fresh validation. Until
then, `batch_serving` is documentation of a deferred topic, not a profile users
should expect to enable.

## Experimental Methods Not In The Default Path

The following methods may become profile cards later, but should not be default
runtime paths without model-specific success-rate evidence:

- PAB / Pyramid Attention Broadcast.
- FasterCache.
- SageAttention.
- xDiT multi-GPU backends.
- TaylorSeer.
- token merging / ToMeSD.
- AsymRnR.
- Sparse VideoGen.
- Q-DiT, PTQ4DiT, ViDiT-Q, and other PTQ methods.
- FP8 TensorRT or Transformer Engine paths.
