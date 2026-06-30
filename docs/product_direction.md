# Product Direction: Systems-Level Deployment And Acceleration Harness

EazyWAM is a systems-level deployment and acceleration harness for WAM/VLA
inference. It should make WAMs understandable, runnable, extensible, and
verifiable across portable backend runtimes.

The primary users are WAM systems researchers and acceleration researchers who
need clear extension points, explicit runtime controls, trustworthy traces, and
comparable validation results. The secondary user is a model runner who wants to
discover, prepare, run, eval, or serve a curated WAM before becoming an expert
in every upstream repository.

## Positioning

EazyWAM is not a benchmark wrapper, training framework, simulator framework, or
universal WAM architecture. It should provide an Ollama-like command experience
for model entries and acceleration profiles:

```bash
wam list
wam info fastwam-libero
wam doctor fastwam-libero
wam prepare fastwam-libero
wam run fastwam-libero --input obs.json --output action.json
wam eval fastwam-libero --workload libero-single-task --opt scheduler --set num_inference_steps=6
wam serve fastwam-libero
wam compare runs/baseline runs/variant
```

The closest analogy is Ollama, but WAMs are harder than LLMs because a usable WAM
entry includes more than weights:

- checkpoint files.
- dataset statistics or normalizers.
- camera and state conventions.
- action schema and control frequency.
- processor logic.
- optional simulator dependencies.
- backend-specific runtime assumptions.

The project wins if it makes this bundle feel like one curated model entry
without hiding the backend and validation boundaries that acceleration work
needs.

## Two Public Catalogs

The model library and acceleration method catalog are different product
surfaces:

- The model library lists curated WAM model entries, their maturity status,
  assets, runtime requirements, and known gaps.
- The acceleration method catalog lists backend-integrated methods, their
  optimization profiles, enablement controls, compatibility, rollout status,
  and validation evidence.

`fastwam-libero` is the first deep exemplar because it currently has the most
complete real WAM path, but it is not privileged architecture. Other model
entries should eventually reach the same maturity standard where their
reasonable run, serve, eval, trace, compare, and acceleration capabilities are
explicit.

Only acceleration methods with `measured` evidence should be promoted as proven
speedups. Implemented or experimental methods can be documented as available,
testable, planned, or unsupported, but not as established wins.

The primary heavy-deployment abstraction is the `wam` command inside a prepared
backend runtime, not a specific cluster. A local container, self-managed virtual
environment, existing GPU allocation, or lab script should all reduce to the
same public command surface. Cluster submission is site policy and stays outside
the core project.

## Two Layers

### Deployment Spine

This is the default user path:

1. Resolve a model id to a curated model entry.
2. Prepare or locate model assets.
3. Load the registered backend and processor.
4. Run explicit-observation or resident server inference.
5. Apply explicit optimization profiles when requested.
6. Wrap the same backend and processor paths in eval workloads when measuring
   task success.
7. Emit runtime info and minimal trace metadata.

The deployment spine optimizes for low friction.

### Telemetry Layer

This is the product observability path:

1. Record where time and memory go.
2. Record active optimization profiles and their parameters.
3. Persist action/future/value artifacts when requested.
4. Compare two recorded runs with `wam compare`.
5. Publish known speed, memory, compatibility, and output-drift notes in the
   support matrix.

`wam compare` is a trust tool, not a marketing badge. It only reports
`faster`/`slower` when both traces have comparable action shapes and usable
latency samples; missing output gates are `not_comparable`.

The telemetry layer optimizes for operational trust. It should not make the
simple run path hard to use.

## What Changes From The Earlier Branch

- The product spine is now `wam list` / `wam info` / `wam doctor` /
  `wam prepare` / `wam run` / `wam serve`.
- The old literature-reproduction workflow has been removed from this branch.
- A curated model library is a goal, but exhaustive model coverage is not.
- Local serving is a goal, but real robot safety and hardware orchestration are
  not first-version goals.
- Model entries become the main default layer, similar in spirit to Ollama's
  model metadata and Modelfile defaults. The internal YAML may still be called a
  manifest in code, but user-facing docs should prefer "model entry" or
  "model spec."

## First Vertical Slices

`A: portable deployment spine`

Rebuild the minimum package: fake model entry, fake backend, open-loop runner,
trace writer, optimization profile metadata, `wam run`, and a container smoke
path.

`B: portable serve smoke`

Start `wam serve fake-open-loop` inside a container or existing job allocation
and run a job-local inference smoke check.

`C: first deep real WAM exemplar`

Make one curated WAM model run deeply through prepare, run, serve, eval, trace,
compare, and acceleration controls. The first deep exemplar is
`fastwam-libero` because it is currently the most mature real WAM path.

`D: first measured acceleration method`

Make one backend-integrated acceleration method toggleable, traceable, and
measurable through an optimization profile. A method can be listed before it is
measured, but only `measured` methods should be presented as proven speedups.

## Differentiation

LeRobot is the likely default place users look for robot learning datasets,
policies, and training workflows. EazyWAM should not compete by being a
larger robot-learning framework.

The wedge is:

- one-command WAM inference deployment.
- curated defaults for known model/checkpoint/task bundles.
- explicit, composable inference optimization profiles.
- a first-class acceleration method catalog alongside the model library.
- trace-backed telemetry for latency, memory, output drift, and compatibility.

In short: use Hugging Face Hub and upstream repos as sources of assets; provide
the deployment and optimization wrapper users wish those repos had.
