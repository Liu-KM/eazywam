# Eval Batching Implementation Plan

> **For agentic workers:** Implement this plan task by task. Keep commits small,
> run the listed tests after each phase, and do not mix simulator sharding with
> model request batching in one unreviewed change.

**Goal:** Add VLA-style eval batching to EazyWAM so multiple simulator episodes
can share one model runtime and issue batched GPU inference requests.

**Architecture:** Follow the vla-eval pattern: episode/env parallelism produces
concurrent action requests, while a backend-owned batch dispatcher dynamically
groups those requests before model inference. The first version should prove the
queue/server/trace contract with fake backends and sharded eval plumbing before
opening true FastWAM `infer_batch()`.

**Tech Stack:** Python stdlib concurrency, EazyWAM `Invocation` /
`BackendSession`, native eval runners, JSONL trace, FastWAM native backend.

**Current status:** Implemented through Phase 5 for the LIBERO smoke case.
Local verification passes, and SuperPod evidence shows that two LIBERO shards
can share one resident FastWAM server with true `fastwam_batch_size=2` requests.

---

## Summary

Eval batching has two separate layers:

- **Episode sharding:** run multiple simulator episodes or shards in separate
  processes so env waiting does not serialize the whole eval.
- **Model request batching:** collect multiple observations at replan points and
  call the model once with a batch, then split actions back to each episode.

The product target is:

```text
multiple simulator shards -> one resident EazyWAM model server/session
-> dynamic batch queue -> backend.infer_batch()
-> per-episode action chunks
```

## Non-Goals For L1

- Do not batch inside one MuJoCo/GL process with threads.
- Do not require FastWAM true batch support before the server contract exists.
- Do not enable CUDA Graph for dynamic batch sizes in L1.
- Do not rewrite LIBERO/RoboTwin managers wholesale.
- Do not change the core observation/action contract for single-request infer.

## Target CLI Shape

```bash
wam serve fastwam-libero \
  --batch \
  --max-batch-size 8 \
  --max-wait-time 0.05

wam eval fastwam-libero \
  --workload libero-single-task \
  --num-trials 50 \
  --num-shards 8 \
  --batch-endpoint http://127.0.0.1:8000
```

Later, a convenience launcher can combine both:

```bash
wam eval-batch fastwam-libero \
  --workload libero-single-task \
  --num-trials 50 \
  --num-shards 8 \
  --max-batch-size 8 \
  --max-wait-time 0.05
```

## Trace Contract

Add trace events or fields that make batching auditable:

- `batch_request_enqueued`
- `batch_dispatch_start`
- `batch_dispatch_end`
- `batch_id`
- `request_id`
- `batch_size`
- `queue_wait_ms`
- `dispatch_ms`
- `per_request_model_ms`
- `batch_fallback_reason`
- `shard_id`
- `num_shards`
- `episode_id`

## Phase 1: Batch Dispatcher Contract

**Goal:** Prove dynamic batching without touching real simulators.

**Files:**

- Create: `src/eazywam/core/batching.py`
- Modify: `src/eazywam/core/registry.py`
- Modify: `src/eazywam/core/backend_session.py`
- Modify: `src/eazywam/serve.py`
- Test: `tests/test_batching.py`
- Test: `tests/test_serve.py`

**Tasks:**

- [x] Add an optional `Backend.infer_batch(requests)` protocol method via
  capability detection, not as a hard requirement.
- [x] Add `BatchDispatcher` with `max_batch_size`, `max_wait_time`, and
  per-request futures.
- [x] If backend has no `infer_batch`, dispatch by looping `infer()` and record
  `batch_fallback_reason="infer_batch_unavailable"`.
- [x] Add `BackendSession.infer_batch_and_trace(...)`.
- [x] Add `wam serve --batch --max-batch-size --max-wait-time`.
- [x] Test that N concurrent fake requests form one batch when possible.
- [x] Test fallback behavior when only `infer()` exists.

**Acceptance:**

```bash
uv run pytest -q tests/test_batching.py tests/test_serve.py
uv run ruff check .
```

Expected: tests pass and trace contains batch queue/dispatch metadata.

## Phase 2: Remote Batch Client For Native Eval

**Goal:** Allow eval workers to ask a resident `wam serve --batch` instance for
actions instead of loading their own model.

**Files:**

- Create: `src/eazywam/core/batch_client.py`
- Modify: `src/eazywam/evals/libero.py`
- Modify: `src/eazywam/evals/robotwin.py`
- Modify: `src/eazywam/cli.py`
- Test: `tests/test_eval_runner.py`

**Tasks:**

- [x] Add a small HTTP client that sends one observation to `/infer` and
  receives an action chunk.
- [x] Add eval overrides `batch_endpoint`, `shard_id`, and `num_shards`.
- [x] In native eval runners, if `batch_endpoint` is set, use the remote client
  at replan points instead of local `Invocation.session.infer_and_trace()`.
- [x] Keep local single-process eval behavior unchanged when no endpoint is set.
- [x] Add trace fields for `shard_id`, `num_shards`, and remote request timing.

**Acceptance:**

```bash
uv run pytest -q tests/test_eval_runner.py tests/test_serve.py
uv run ruff check .
```

Expected: local eval tests still pass; remote-client tests prove request routing
without loading a local backend in each shard.

## Phase 3: Episode Sharding

**Goal:** Split episodes across independent OS processes.

**Files:**

- Modify: `src/eazywam/evals/libero.py`
- Modify: `src/eazywam/evals/robotwin.py`
- Modify: `src/eazywam/core/eval_runner.py`
- Modify: `src/eazywam/cli.py`
- Test: `tests/test_eval_runner.py`
- Test: `tests/test_fastwam_robotwin_manager.py`

**Tasks:**

- [x] Define shard assignment: `episode_idx % num_shards == shard_id`.
- [x] Add dry-run output showing selected episode indices per shard.
- [x] Ensure per-shard output directories include `shard-{id}`.
- [x] Add a merge summary helper for shard result JSON files.
- [x] Do not run multiple simulator envs in one Python process for L1.

**Acceptance:**

```bash
uv run pytest -q tests/test_eval_runner.py tests/test_fastwam_robotwin_manager.py
uv run ruff check .
```

Expected: shard dry-run is deterministic, merge summaries count requested,
completed, failed, and skipped episodes.

## Phase 4: FastWAM True `infer_batch()`

**Goal:** Replace loop fallback with actual batched FastWAM inference for small
batch sizes.

**Files:**

- Modify: `src/eazywam/backends/fastwam.py`
- Modify: `src/eazywam/processors/fastwam_libero.py`
- Modify: `src/eazywam/processors/fastwam_robotwin.py`
- Modify: `src/fastwam/models/wan22/fastwam.py`
- Test: `tests/test_fastwam_native.py`

**Tasks:**

- [x] Add processor batch packing for homogeneous observations.
- [x] Add processor batch unpacking into per-request action chunks.
- [x] Add FastWAM adapter `infer_batch()` for `batch_size=2` first.
- [x] Validate action shape parity between `infer()` and `infer_batch()`.
- [x] Disable CUDA Graph by default on dynamic batch path.
- [x] Add batch metadata: `fastwam_batch_size`, `batch_shape_key`,
  `batch_cuda_graph_enabled=false`.

**Acceptance:**

```bash
uv run pytest -q tests/test_fastwam_native.py tests/test_batching.py
uv run ruff check .
```

Expected: fake and FastWAM adapter tests pass; true model path is ready for
SuperPod smoke.

## Phase 5: SuperPod Evidence

**Goal:** Demonstrate throughput improvement with real simulator eval.

**Runs:**

- LIBERO single task, same task/seed/trials:
  - serial local eval
  - sharded eval + batch server with loop fallback
  - sharded eval + true `infer_batch()`
- RoboTwin single task, same task/seed/episodes:
  - serial local eval
  - sharded eval + batch server with loop fallback
  - sharded eval + true `infer_batch()`

**Metrics:**

- episodes/hour
- observations/sec produced by env shards
- requests/sec consumed by model server
- mean and p90 queue wait
- mean batch size
- GPU memory
- success rate
- per-request latency

**Acceptance:**

- Batch path keeps success rate within the serial baseline for smoke tasks.
- Trace proves batches actually formed (`batch_size > 1`).
- Report separates env-sharding speedup from true model-batching speedup.

**Maintainer evidence:** SuperPod H800 run `eval-batching-batch-453372`
validated FastWAM LIBERO task 0 with `num_trials=2`, `num_shards=2`,
`max_batch_size=2`, and `max_wait_time=1.0`. It reused the serial baseline from
run `eval-batching-453363`.

- Serial native eval: 2/2 successes, 148 wall seconds, 48.65 episodes/hour.
- Sharded batch-server eval: 2/2 successes, 58 wall seconds, 124.14 episodes/hour.
- Wall-clock speedup: 2.55x on this small smoke.
- Server trace: 59 `/infer` requests, 35 dispatches, 24 dispatches with
  `batch_size=2`, mean batch size 1.69, no batch fallback reasons.
- FastWAM metadata observed `fastwam_batch_size=2`,
  `batch_cuda_graph_enabled=false`, and `batch_shape_key.input_image=[2,3,224,448]`.

This is smoke evidence, not a final throughput claim. The next measurement
should sweep `num_trials`, `num_shards`, and `max_wait_time` after the CLI flow
is merged.

## Final Completion Criteria

- `wam serve --batch` works and is traced.
- Native eval can target a resident batch endpoint.
- LIBERO and RoboTwin can shard episodes across processes.
- FastWAM has a real `infer_batch()` path or a clearly traced fallback.
- SuperPod evidence reports both throughput and success rate.
- Full local verification passes:

```bash
uv run pytest -q
uv run ruff check .
```
