# EazyWAM

EazyWAM is an open-source deployment and acceleration platform for
world-action model inference. This context defines the product language used by
the codebase, documentation, issues, and future design discussions.

## Language

**EazyWAM**:
The formal product name for the systems-level deployment and acceleration
harness for world-action models. EazyWAM turns each WAM into a curated runnable
model entry and gives each acceleration method an isolated implementation path,
explicit enable/disable controls, traceable runtime metadata, and comparable
evaluation results. The public command is `wam`; local repository paths such as
`wam-harness` are not product names.
_Avoid_: WAM Harness, wam-harness

**Acceleration Method**:
A backend-integrated inference-time or runtime technique intended to make WAM
execution faster, cheaper, or more deployable. Each method needs source code, a
clear hook point, unified CLI or runtime controls, declared parameters, scope,
compatibility rules, runtime status reporting, tests, and documentation.
_Avoid_: Trick, hidden optimization

**Optimization Profile**:
The EazyWAM contract for enabling, configuring, tracing, and comparing an
acceleration method. A profile is not the method's core implementation; it is
the user-facing and runtime-facing control surface for that method.
_Avoid_: Acceleration method implementation

**Opt Flag**:
The shared public CLI mechanism for enabling an acceleration method on run,
eval, or serve paths: `--opt <method-name>`. Advanced method parameters should
use shared overrides such as `--set key=value` or request runtime options rather
than backend-specific public flags.
_Avoid_: Backend-specific acceleration flag

**WAM Systems Researcher**:
The primary EazyWAM user: someone developing or evaluating WAM deployment,
runtime, serving, or inference-acceleration methods. They need clear extension
points for new acceleration methods and trustworthy traces and comparisons.
_Avoid_: Benchmark-only user

**Model Runner**:
The secondary EazyWAM user: someone who wants to discover, prepare, run, eval,
or serve a curated WAM without first understanding every upstream repository.
_Avoid_: Framework maintainer

**Core Contract**:
The system boundary EazyWAM standardizes across WAMs: model entries, backend
lifecycle, processor conversion, workloads, optimization profile controls,
traces, and comparisons. It does not define a universal internal algorithm
interface for WAMs.
_Avoid_: Model architecture abstraction

**Backend-Native Execution Path**:
The model-specific path inside a backend where an upstream WAM actually loads,
samples, caches, denoises, serves, or evaluates. Acceleration methods may modify
this path, while EazyWAM standardizes how those methods are selected,
configured, traced, and compared.
_Avoid_: Core runner logic

**Backend**:
The EazyWAM module that owns model execution and runtime state for a WAM. A
backend loads, warms up, resets, infers, closes, reports runtime information,
and applies execution-path acceleration methods.
_Avoid_: Processor, workload

**Processor**:
The EazyWAM module that owns semantic I/O translation between EazyWAM
observations/results and backend-native inputs/outputs. A processor handles
image view selection, prompt formatting, state mapping, action conversion,
modality limits, and synthetic smoke observations.
_Avoid_: Backend, model runner

**Validation Path**:
The shared EazyWAM path for checking whether a model run or acceleration method
is trustworthy. Validation uses traces, comparisons, eval metrics, runtime
status, and documented measurement results; it is a platform capability, not
part of any one acceleration method's source implementation.
_Avoid_: Method implementation

**Acceleration Validation Loop**:
The public EazyWAM workflow for proving the status of an acceleration method:
declare a baseline, run a variant through `--opt` or an equivalent profile,
record traces and summaries, compare outputs and runtime metrics, check eval
quality when relevant, record fallback status, and publish scoped evidence.
Maintainers may execute this loop on the SuperPod Maintainer Runtime during
development, but the loop must be expressible for any prepared GPU runtime and
must not depend on SuperPod-specific launch mechanics. The first public runbook
should focus on `wam run`, `wam eval`, and `wam compare`; `wam serve`
acceleration validation is a required follow-up track for resident runtime,
warmup, request-latency, health, long-running stability, and batching behavior.
_Avoid_: SuperPod-only benchmark flow

**Acceptance Validation**:
The first validation level for an acceleration method. It proves that the method
is selectable, trace-visible, contract-compatible, and able to complete at least
one relevant small workload without obvious task-quality failure. Acceptance
validation can be produced by the same run that later supports measured
validation, but it answers "is this method usable?" rather than "is this a
reportable speedup?"
_Avoid_: Speedup proof

**Measured Validation**:
The second validation level for an acceleration method. It proves a scoped
speedup or memory improvement by comparing a clear baseline and variant under a
declared model, workload, runtime, seed/trial scope, fallback policy, task
quality gate, and reportable performance metric. Only measured validation can
promote an acceleration method to scoped `measured` status.
_Avoid_: Smoke test, anecdotal benchmark

**Public Command Surface**:
The stable user-facing command set: `wam list`, `wam info <model-id>`,
`wam doctor <model-id>`, `wam prepare <model-id>`, `wam run <model-id>`,
`wam eval <model-id>`, `wam serve <model-id>`, and `wam compare <run-a>
<run-b>`. Acceleration methods are enabled through shared options such as
`--opt <method>` on run, eval, or serve commands.
_Avoid_: One-off scripts as primary workflow

**Run Path**:
The one-shot EazyWAM product path that turns one explicit observation into one
action result through the model entry, processor, backend, optimization profile
controls, and trace contract.
_Avoid_: Synthetic benchmark path

**Serve Path**:
The resident EazyWAM product path that keeps a backend loaded and handles
repeated observation-to-action requests through the same model entry, processor,
backend, optimization profile controls, and trace contract as the run path.
_Avoid_: External endpoint as the only product path

**Eval Path**:
The EazyWAM validation path that wraps the same backend and processor used by
run or serve inside a workload or simulator loop to measure task success and
episode metrics.
_Avoid_: Separate default execution path

**Reference Eval**:
An explicit eval mode that calls an upstream official script for parity or
debugging. It is useful for comparison, but it is not the default EazyWAM
product path.
_Avoid_: Default eval path

**Lightweight Core Install**:
The EazyWAM core package must remain installable without pulling in every heavy
WAM stack. CLI, contracts, registry, trace, compare, fake backend, lightweight
adapters, and orchestration belong in core; CUDA-heavy runtimes, simulators, and
upstream-specific dependency stacks belong in backend runtimes or containers.
_Avoid_: Monolithic WAM environment

**Upstream WAM Repository**:
An external WAM project that provides model code, checkpoints, configuration,
processor logic, simulator assumptions, or official scripts. EazyWAM integrates
upstream WAMs through model entries, processors, backends, and reference evals;
it does not copy their architecture into the core harness.
_Avoid_: Core architecture source

**Depth-First Roadmap**:
The product route for EazyWAM: first make one real WAM deeply complete across
prepare, run, serve, eval, trace, compare, and acceleration methods; then use a
second model or task family to test generality; then expand model coverage and
the acceleration method library.
_Avoid_: Many shallow wrappers

**Contribution Path**:
The expected open-source contribution routes: add a model entry, add or improve
a backend/processor integration, or add an acceleration method with a clear hook
point, `--opt` control, parameters, scope, compatibility rules, runtime status,
tests, documentation, and validation through shared EazyWAM paths.
_Avoid_: One-off benchmark script

**README Positioning**:
The first screen of the README should present EazyWAM as a systems-level
deployment and acceleration harness for WAM/VLA inference, aimed first at WAM
systems and acceleration researchers while keeping the command experience simple
for model runners.
_Avoid_: Generic model runner only

**Open-Source Acceleration Harness Roadmap**:
The next PRD focus for EazyWAM. It should make the project understandable,
runnable, extensible, and verifiable for outside WAM systems and acceleration
researchers by aligning public docs, deepening one real WAM exemplar, defining
the acceleration method contribution path, hardening validation, and planning
generality and library expansion.
_Avoid_: Documentation cleanup only

**Deep Exemplar**:
A currently mature model entry used to demonstrate the full EazyWAM path across
prepare, run, serve, eval, trace, compare, and acceleration methods. The first
deep exemplar is `fastwam-libero`, but it is not privileged architecture; other
models should be brought to the same level over time.
_Avoid_: Architecture center, special-case model

**Full Model Entry Maturity**:
The target completeness level for a supported model entry. A mature entry has
clear info, doctor, prepare, run, serve, eval, trace, acceleration method
support declarations, validation status, and documentation for the capabilities
that model can reasonably support; unsupported or unvalidated acceleration
methods must be explicit.
_Avoid_: Every model supports every method

**Model Library**:
The public catalog of supported WAM model entries and their maturity status.
It helps users discover which WAMs can be prepared, run, evaluated, served, and
optimized through EazyWAM.
_Avoid_: Raw upstream repo list

**Acceleration Method Catalog**:
A first-class EazyWAM catalog of supported, planned, experimental, and measured
acceleration methods. It explains what each method is, how to enable it, which
models can support it, and what validation status exists.
_Avoid_: Incidental optimization notes

**Acceleration Method Status**:
The lifecycle label for an acceleration method on a model or backend:
`planned`, `implemented`, `experimental`, `measured`, or `unsupported`. Only
`measured` methods should be promoted as proven speedups; implemented or
experimental methods may be documented as available or testable, not proven.
Passing acceptance validation usually promotes a method to `experimental`, not
`measured`; `measured` requires measured validation in a declared scope.
_Avoid_: Marketing unmeasured speedups

**Measured Method Evidence**:
The minimum evidence required before an acceleration method is labeled
`measured` for a model entry: model and workload, baseline command, variant
command, trace path or trace summary, latency or memory comparison, action shape
gate, output drift summary when comparable, simulator success or episode metrics
when relevant, fallback status, and hardware/runtime environment. Output drift
is an audit signal by default; task quality is the primary gate for simulator
workloads unless a method explicitly claims exact output preservation.
_Avoid_: Anecdotal speedup claim

**Public Evidence Summary**:
The open-source-safe evidence record for a model validation or acceleration
method result. It includes the scoped status conclusion, model, workload, seed,
trial count, hardware class, dtype, runtime, baseline command, variant command,
trace/report summary, compare summary, output drift summary when comparable,
fallback status, task-quality result when relevant, and the reason a method is
or is not reportable as `measured`. It excludes private scheduler wrappers,
accounts, partitions, login details, scratch paths, raw logs, credentials, and
site recovery scripts. Public validation runbooks should describe generic GPU
runtime requirements and standard `wam` commands rather than naming private
maintainer platforms; a private platform may appear only as scoped evidence
context when relevant.
_Avoid_: Raw cluster log dump, private SuperPod runbook

**Evidence Summary**:
The short human-readable review layer for an acceleration or model-validation
result. It states the status decision, scope, baseline, variant, task quality,
performance result, profile/fallback status, output-drift audit result when
available, and caveats in a form a maintainer can review quickly. It should not
force humans to read a large machine JSON object before understanding the
claim.
_Avoid_: Machine-only evidence dump

**Evidence Bundle**:
The complete machine-readable evidence package used by agents, tools, CI, and
future maintainers. It contains the Evidence Summary plus structured JSON,
compare output, trace or trace-summary references, commands, runtime context,
profile status, acceptance validation, measured validation, blockers, and
privacy/redaction notes. The bundle is intentionally more detailed than the
human summary so status decisions can be validated automatically later.
_Avoid_: Unstructured benchmark notes

**SuperPod Maintainer Runtime**:
The private maintainer GPU/simulator environment used by EazyWAM maintainers to
produce real acceleration and model-validation evidence. SuperPod can appear in
evidence records as hardware/runtime context, but it is not a public EazyWAM
deployment interface, not an open-source user requirement, and not part of the
core command contract. SuperPod operational knowledge such as sbatch files,
accounts, partitions, scratch paths, recovery scripts, and login workflows
should stay in untracked local maintainer notes, not public repository docs.
_Avoid_: Public deployment target, required user platform

**Acceleration Method Developer Guide**:
A core open-source deliverable that explains how to add a new acceleration
method: choose the model/backend, find the backend-native hook point, implement
the method, expose it through `--opt` and runtime options, declare parameters
and conflicts, report runtime status, add trace fields, test it, validate it,
and update the acceleration method catalog.
_Avoid_: Read the code and guess

**Parallel Roadmap Tracks**:
The implementation style for the next roadmap PRD. Work should be split into
parallelizable tracks such as public docs, contribution experience,
acceleration catalog and validation, deep exemplar, and architecture alignment;
dependencies between issues must be explicit instead of hidden in a single
serial milestone.
_Avoid_: One long linear task list

**Model Entry**:
The standard EazyWAM definition for one WAM. It tells the system how to
prepare, run, eval, serve, and optimize that model by naming its model id,
assets, backend, processor, workloads, defaults, supported optimization
profiles, and known gaps.
_Avoid_: Raw YAML, weight file
