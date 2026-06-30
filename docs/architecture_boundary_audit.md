# Architecture Boundary And Dependency Isolation Audit

Date: 2026-06-30

This audit checks whether the current implementation follows the EazyWAM
boundary decisions:

- core code reasons in harness contracts, not specific WAM repository internals;
- manifests map model ids to backends, processors, assets, defaults, workloads,
  and optimization profiles;
- backends own model loading, connection, lifecycle, and native inference;
- processors own observation packing and result unpacking;
- heavy runtime dependencies stay out of the core install path;
- reference evals remain explicit and separate from product evals.

## Files Inspected

- `src/eazywam/core/runner.py`
- `src/eazywam/core/registry.py`
- `src/eazywam/core/invocation.py`
- `src/eazywam/core/runtime.py`
- `src/eazywam/cli.py`
- `src/eazywam/defaults.py`
- `src/eazywam/backends/*`
- `src/eazywam/processors/*`
- `docs/runtime_abstraction.md`
- `docs/dependency_isolation.md`
- `docs/optimization_profiles.md`
- `docs/acceleration_methods.md`

## Findings

| Area | Result | Notes |
|---|---|---|
| Core runner | Pass with one cleanup | `Runner` operates on `Manifest`, `Invocation`, workloads, traces, action horizon, and replan controls. It does not load FastWAM, Cosmos-Policy, or DreamZero directly. The input-required hint was made model-generic instead of naming a LIBERO workload from core. |
| Registry | Pass | The registry owns factories for manifests, backends, processors, workloads, eval runners, and optimization defaults. Model-specific classes enter through registration, not through core conditional branches. |
| Manifest/runtime resolution | Pass | Runtime plans transform a curated model entry into a backend/processor/workload path without requiring users to hand-write a large capabilities file. |
| CLI | Acceptable | CLI help includes real model examples for discoverability. That is user-facing guidance, not control flow. CLI commands still dispatch through model ids and registry-backed contracts. |
| Backends | Expected model specificity | FastWAM, Cosmos-Policy, and DreamZero details live under `src/eazywam/backends/`. This is the intended place for upstream imports, model adapters, runtime loaders, profile hooks, and native error messages. |
| Processors | Expected model specificity | Observation layout, state keys, image conversion, action chunk conversion, and modality limits live under `src/eazywam/processors/`. These details should not move into core runner or compare logic. |
| Optimization profiles | Pass | Profiles are selected by stable names such as `scheduler`, `dit_cache`, `cuda_graph`, and `teacache`. Backend-specific hooks are trace-visible metadata, not public CLI flags. |
| Dependency isolation | Pass for core | Core modules do not import torch, CUDA, MuJoCo, LIBERO, RoboTwin, JAX, or upstream WAM packages. Heavy imports occur inside backend/runtime loader paths and should run only in compatible backend environments. |
| Reference eval boundary | Pass | `wam eval --reference` is explicit and separate from native product eval. Reference traces should contain `external_eval_plan` / `external_eval_end`, while product eval traces should contain native eval events. |

## Boundary Rules For Future Issues

- Do not add `if model_id == ...` or repository-name branches to
  `src/eazywam/core/*`.
- Do not import torch, simulator packages, or upstream WAM repos from core
  package modules.
- Do not expose backend-native tensor layouts, cache mechanics, scheduler
  object names, or transport details as required core interfaces.
- Do not add model-specific public CLI flags when `--opt`, `--set`,
  `--backend-set`, manifests, or `runtime_options` can express the same choice.
- Do not treat official upstream eval scripts as the default product eval path.
  Keep them behind `--reference`.
- Do not record a speedup claim unless `docs/acceleration_methods.md` measured
  evidence requirements are met.
- User-facing examples may name real models, but control flow must stay
  registry-driven.

## Known Follow-Ups

- Add a lightweight boundary test that scans `src/eazywam/core` for forbidden
  heavy imports and accidental model-id branches.
- Keep backend files from growing into unreviewable monoliths by extracting
  runtime loaders, model adapters, and profile hooks only when the split reduces
  real complexity.
- Keep `docs/fastwam_libero_deep_exemplar.md` as the template for future deep
  model entries so FastWAM does not become an implicit special case.
- Revisit plugin or external registry loading once the built-in manifest and
  backend list becomes too large for one default registry module.
