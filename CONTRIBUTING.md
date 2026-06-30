# Contributing

EazyWAM is a systems-level deployment and acceleration harness for WAM/VLA
inference. Contributions should preserve the small `wam` command surface while
making model entries, backends, processors, optimization profiles, traces, and
validation easier to extend.

Before starting, read `AGENTS.md`, `docs/product_direction.md`,
`docs/contract.md`, and the issue or PRD that defines your task.

## Contribution Paths

Most contributions should fit one of these paths.

### Model Entry

Use this path when adding or maturing a curated model id such as
`fastwam-libero`.

A model entry contribution should include:

- model id, display name, upstream source, license/provenance, and maturity
  status.
- required checkpoint, normalizer, dataset-stat, simulator, or asset metadata.
- backend key, processor key, workload/eval declarations, and curated defaults.
- supported, unsupported, experimental, and measured acceleration profile
  declarations.
- known gaps, hardware/runtime requirements, and prepare/doctor notes.
- tests for parsing, registry resolution, supported optimization profiles, and
  any fake or lightweight smoke path.

Do not add heavy WAM, simulator, CUDA, or cluster dependencies to the core
package just to make a model entry importable. Heavy stacks belong in backend
runtimes, containers, external checkouts, or explicit prepare/doctor flows.

### Backend Or Processor Integration

Use this path when connecting EazyWAM to a model runtime or translating
observations/results for a model family.

Backends own execution and lifecycle:

- `load`, `warmup`, `reset`, `infer`, `runtime_info`, and `close`.
- native process/server startup and cleanup.
- backend-owned optimization hooks and fallback metadata.
- runtime information such as model id, backend, processor, mode, device, dtype,
  and active optimization profiles.

Processors own semantic I/O translation:

- image view selection and preprocessing.
- prompt formatting and state-vector mapping.
- action denormalization and output artifact conversion.
- synthetic smoke observations, modality limits, and input requirements.

Keep backend-native tensor layouts, cache internals, scheduler calls, transport
protocols, and upstream repository control flow behind the backend or processor
boundary. The core runner should reason in observations, requests, results,
runtime info, and traces.

### Acceleration Method

Use this path when adding or validating a runtime acceleration method.

An acceleration method contribution should include:

- source code or backend adapter code at the real hook point.
- a stable optimization profile name exposed through `--opt <method>` on
  `wam run`, `wam eval`, or `wam serve` where applicable.
- typed parameters, defaults, compatibility rules, conflicts, and fallback
  behavior.
- runtime status: `planned`, `implemented`, `experimental`, `measured`, or
  `unsupported`.
- trace fields and `wam compare` behavior for latency, memory, output drift,
  action-shape gates, fallback reasons, and eval metrics when relevant.
- docs in `docs/acceleration_methods.md` and `docs/optimization_profiles.md`.
- tests for CLI/profile construction, backend option propagation, trace
  metadata, fallback status, and compare gates.

Do not contribute a one-off benchmark script as the primary product path. A
script can help generate evidence, but the method should still be selectable
through an EazyWAM profile and validated through the product run/eval/serve
paths.

Only describe a method as a proven speedup when it reaches `measured` status
with the evidence required by `docs/acceleration_methods.md`.

## Validation

Use `uv` for core development:

```bash
uv sync --dev
uv run ruff check .
uv run pytest
```

Docs-only changes should still check public commands, links, status tables, and
terminology. If user-facing README content changes, update `README.md` and
`README.zh-CN.md` together.

## Useful References

- `docs/acceleration_methods.md` - acceleration method catalog and measured
  evidence standard.
- `docs/optimization_profiles.md` - profile taxonomy and profile-card contract.
- `docs/trace_schema.md` - trace events and profile metadata.
- `docs/cli_entrypoints.md` - public command behavior.
- `docs/wamfile.md` - model entry schema.
- `docs/runtime_abstraction.md` - runtime info and backend boundaries.
- `docs/dependency_isolation.md` - lightweight core and heavy runtime isolation.
