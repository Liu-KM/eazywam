# EazyWAM Open-Source Acceleration Harness Roadmap PRD

## Problem Statement

EazyWAM has grown from a deployment-spine skeleton into a real WAM/VLA
deployment and acceleration codebase, but its public story, contribution paths,
and next development plan are not yet clear enough for outside WAM systems and
acceleration researchers.

Today, a new contributor can see model entries, backend integrations, eval
paths, traces, and acceleration profiles, but they still have to infer the
project's core product shape: EazyWAM is not a benchmark wrapper, not a training
framework, and not a FastWAM-only harness. It is a systems-level deployment and
acceleration harness where WAMs become standard model entries and acceleration
methods become isolated, controllable, traceable, and comparable runtime
integrations.

Without a clear PRD and issue breakdown, follow-on agents may keep improving
isolated files while missing the larger roadmap: make the project understandable,
runnable, extensible, and verifiable for researchers who want to run WAMs and
develop new acceleration methods.

## Solution

Make the next development stage an open-source acceleration harness roadmap.
This roadmap aligns public docs, contribution paths, acceleration method
catalogs, validation standards, and the first deep real WAM exemplar.

The roadmap should produce five parallel tracks:

1. Public positioning and documentation alignment.
2. Contribution experience for model entries, backend/processors, and
   acceleration methods.
3. Acceleration method catalog and validation rules.
4. A deep `fastwam-libero` exemplar that demonstrates the full EazyWAM path.
5. Architecture and dependency-boundary alignment.

The first deep exemplar is `fastwam-libero` because it is the most mature
current real WAM path, but it is not privileged architecture. Other supported
models should eventually reach the same maturity level where their reasonable
capabilities are documented, runnable, validated, and explicit about unsupported
or unverified acceleration methods.

## User Stories

1. As a WAM systems researcher, I want the README to explain EazyWAM as a
   systems-level deployment and acceleration harness, so that I know whether the
   project is relevant to my work.
2. As a WAM acceleration researcher, I want to see where a new acceleration
   method should hook into a backend, so that I can start modifying the right
   part of the system.
3. As a WAM acceleration researcher, I want acceleration methods to have a
   shared `--opt` entry point, so that I do not need to learn model-specific
   public flags for every method.
4. As a WAM acceleration researcher, I want each acceleration method to declare
   parameters, scope, compatibility, conflicts, and runtime status, so that I can
   understand whether it applies to my target model.
5. As a WAM acceleration researcher, I want `implemented`, `experimental`,
   `measured`, and `unsupported` statuses to mean specific things, so that I do
   not confuse available code with proven speedups.
6. As a WAM acceleration researcher, I want measured methods to include baseline
   commands, variant commands, trace summaries, eval metrics, fallback status,
   and hardware context, so that I can judge whether the evidence is useful.
7. As a WAM acceleration researcher, I want a guide for adding acceleration
   methods, so that I can follow the expected path from source implementation to
   validation.
8. As a model runner, I want the command surface to stay simple, so that I can
   discover, prepare, run, eval, and serve a model without understanding every
   upstream repository.
9. As a model runner, I want `wam info` and documentation to explain known gaps,
   assets, hardware requirements, and supported methods, so that I can decide
   whether a model entry is usable for me.
10. As a contributor adding a model entry, I want a clear maturity standard, so
    that I know what is required beyond adding a YAML file.
11. As a contributor adding a backend, I want backend and processor boundaries
    to be explicit, so that I do not leak tensor layouts or upstream control flow
    into the core runner.
12. As a contributor adding a processor, I want to know that processors own
    semantic I/O translation, so that image views, prompt formatting, state
    mapping, and action conversion stay in the right place.
13. As a maintainer, I want upstream WAM repositories treated as integration
    targets rather than core architecture sources, so that the project remains
    extensible across heterogeneous WAMs.
14. As a maintainer, I want the core install to remain lightweight, so that
    `pip install eazywam` does not pull in FastWAM, Cosmos-Policy, DreamZero,
    simulators, or CUDA-heavy dependency stacks.
15. As a maintainer, I want the roadmap split into parallel tracks, so that
    multiple agents can work without being blocked by a single serial plan.
16. As an agent implementing an issue, I want dependencies called out explicitly,
    so that I can tell whether my task can start immediately.
17. As an agent implementing an issue, I want a parent PRD to reference, so that
    I can preserve the project vocabulary and decisions without rereading the
    full planning conversation.
18. As a maintainer, I want `fastwam-libero` to be a deep exemplar but not a
    privileged model, so that useful examples do not turn into architecture
    special cases.
19. As a maintainer, I want the acceleration method catalog to be as visible as
    the model library, so that EazyWAM is understood as an acceleration platform,
    not only a model runner.
20. As a maintainer, I want docs to distinguish acceleration methods from
    optimization profiles, so that contributors understand that a profile is the
    control and validation contract, not the method implementation itself.
21. As a maintainer, I want eval paths to validate the run and serve product
    paths, so that EazyWAM does not regress into a collection of independent
    benchmark wrappers.
22. As a maintainer, I want explicit reference eval language, so that official
    upstream scripts remain useful for parity without becoming the default
    product path.
23. As a maintainer, I want architecture and documentation audits before broad
    feature expansion, so that the next wave of agent work reinforces the right
    system boundaries.

## Implementation Decisions

- EazyWAM is a systems-level deployment and acceleration harness for WAM/VLA
  inference. It should not be described as a benchmark wrapper, a training
  framework, a simulator framework, or a universal WAM model architecture.
- The primary user is the WAM systems researcher or acceleration researcher. The
  secondary user is the model runner who wants simple Ollama-like commands.
- The stable public command surface is `wam list`, `wam info`, `wam doctor`,
  `wam prepare`, `wam run`, `wam eval`, `wam serve`, and `wam compare`.
- Acceleration methods are backend-integrated techniques with source code,
  hook points, shared controls, parameters, scope, compatibility rules, runtime
  status, tests, and documentation.
- Optimization profiles are the EazyWAM control contract for acceleration
  methods. Profiles enable, configure, trace, and compare methods; they are not
  the method implementations themselves.
- The shared public acceleration control is `--opt <method-name>` on run, eval,
  or serve paths. Advanced parameters should use shared overrides or runtime
  options rather than backend-specific public flags.
- Backend owns model execution and runtime state. Processor owns semantic I/O
  translation between EazyWAM observations/results and backend-native
  inputs/outputs.
- EazyWAM core does not define a universal internal WAM algorithm interface.
  Acceleration methods may modify backend-native execution paths, but core
  should not expose tensor layouts, normalization details, cache hooks, or
  upstream repository control flow.
- The core package must remain lightweight. Heavy WAM runtimes, simulators, and
  upstream-specific dependency stacks belong in backend runtimes, containers,
  external checkouts, or explicit prepare/doctor flows.
- Upstream WAM repositories are backend targets and asset sources, not core
  architecture sources.
- Run and serve are the primary EazyWAM product paths. Eval wraps those same
  backend and processor paths inside a workload or simulator loop to measure
  task success. Reference evals may call upstream official scripts, but must
  stay explicit and separate.
- The roadmap is depth-first: first make one real WAM deeply complete, then use
  a second model or task family to test generality, then expand model coverage
  and the acceleration method catalog.
- `fastwam-libero` is the first deep exemplar because it is currently mature,
  but it must not become privileged architecture.
- Mature model entries should expose clear info, doctor, prepare, run, serve,
  eval, trace, acceleration support declarations, validation status, and
  documentation for capabilities that the model can reasonably support.
- The acceleration method catalog is a first-class product surface alongside
  the model library.
- Acceleration method status values are `planned`, `implemented`,
  `experimental`, `measured`, and `unsupported`. Only `measured` methods should
  be promoted as proven speedups.
- Minimum measured evidence includes model and workload, baseline command,
  variant command, trace path or summary, latency or memory comparison, action
  shape gate, simulator success or episode metrics when relevant, fallback
  status, and hardware/runtime environment.
- Work should be split into parallelizable tracks with explicit dependencies:
  public docs, contribution experience, acceleration catalog and validation,
  deep exemplar, and architecture alignment.

## Testing Decisions

- Test through the highest stable seam available. For docs-only changes, verify
  that public command examples, terminology, and status tables remain consistent
  with the existing command surface and model entry contracts.
- For contribution and developer guides, the test is whether an agent can use
  the document to produce a focused implementation issue without relying on
  hidden conversation context.
- For acceleration method catalog changes, verify that every listed method has a
  status, model applicability, enablement path, and validation notes. Avoid
  promoting methods as measured unless the minimum evidence fields are present.
- For deep exemplar work, verify that the documented path covers prepare, run,
  serve, eval, trace, compare, and at least one acceleration method path without
  turning the exemplar into a special-case architecture.
- For backend/processor/core boundary audits, prefer tests that exercise public
  behavior: registry construction, CLI option propagation, trace metadata,
  runtime status events, and action result contracts.
- Continue using the existing core verification commands before claiming
  completion of implementation issues: `uv run pytest` and `uv run ruff check .`.
- Do not require GPU, simulator, or heavy upstream dependencies for documentation
  and core-boundary issues unless the issue explicitly targets a real backend
  validation path.

## Out of Scope

- Rewriting the full codebase before documentation and roadmap alignment.
- Making every model support every acceleration method.
- Adding heavy WAM dependencies to the core package.
- Turning EazyWAM into a training framework, simulator framework, real robot
  control stack, cluster scheduler, or universal tensor runtime.
- Treating upstream official eval scripts as the default product path.
- Publishing unmeasured acceleration methods as proven speedups.
- Expanding the model library with many shallow wrappers before at least one
  deep exemplar and one generality proof are clear.
- Requiring full paper-grade statistical evidence before a method can be
  documented as implemented or experimental.

## Further Notes

The first issue split should include one parent PRD issue followed by parallel
agent-ready issues. Good first slices include README positioning, product
direction and roadmap alignment, contribution paths, acceleration method
developer guide, acceleration method catalog, measured evidence checklist,
`fastwam-libero` deep exemplar checklist, full-path tutorial, and architecture
boundary audit.

Issue dependencies should be explicit. The acceleration method catalog format
must exist before auditing current methods into the catalog. The deep exemplar
checklist must exist before writing the full-path tutorial. Architecture audit
findings should precede boundary fixes. Public docs, contribution docs,
catalog scaffolding, exemplar checklist, and architecture audit can proceed in
parallel after this PRD is accepted.
