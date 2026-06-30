# Acceleration Validation Runbook And Evidence Bundle PRD

## Problem Statement

EazyWAM now treats acceleration methods as a first-class product surface, but
contributors and agents still lack a standard way to prove whether an
acceleration method is merely implemented, experimentally usable, or measured
as a scoped speedup.

Existing evidence is split across traces, compare output, method-specific
reports, local maintainer scripts, and narrative docs. That makes it too easy
to confuse "the code path exists" with "the method is a reportable speedup."
It also risks leaking maintainer-only runtime details into public docs.

The next step is to define a public acceleration validation loop that any
prepared GPU runtime can execute through `wam` commands, while keeping private
maintainer runtime operations out of the open-source product contract.

## Solution

Create a small public runbook and evidence template for acceleration method
validation.

The runbook should define:

- how to run acceptance validation for an acceleration method;
- how to run measured validation for a scoped speedup claim;
- how to compare baseline and variant traces;
- how to write a short human Evidence Summary;
- how to attach a structured Evidence Bundle for agents and future tooling;
- how evidence updates the acceleration method catalog.

The first public runbook should focus on `wam run`, `wam eval`, and
`wam compare`. `wam serve` acceleration validation is a required follow-up
track, but it should not be part of the first runbook's main path.

SuperPod or any other maintainer GPU platform is allowed as an execution
environment for maintainers, but public docs must describe generic prepared GPU
runtimes and standard `wam` commands. SuperPod sbatch files, account names,
partitions, scratch paths, recovery scripts, login workflows, and raw private
logs stay in untracked local maintainer notes.

## User Stories

1. As a WAM acceleration researcher, I want a standard validation loop, so that
   I know what evidence is required before calling a method measured.
2. As a contributor implementing an acceleration method, I want an acceptance
   validation path, so that I can prove the method is selectable,
   trace-visible, and contract-compatible before running expensive experiments.
3. As a contributor implementing an acceleration method, I want a measured
   validation path, so that I can prove a scoped latency or memory improvement
   without overclaiming.
4. As a maintainer, I want status decisions to distinguish `implemented`,
   `experimental`, and `measured`, so that the acceleration method catalog does
   not market unverified speedups.
5. As a maintainer, I want evidence to include baseline and variant commands,
   so that a future reviewer can understand what was compared.
6. As a maintainer, I want evidence to include runtime context, so that hardware
   or dtype differences do not create invalid comparisons.
7. As a maintainer, I want evidence to include profile requested/applied/fallback
   status, so that silent fallback cannot be mistaken for acceleration.
8. As a maintainer, I want output drift recorded as an audit signal, so that
   output changes are visible without making drift the default hard gate for
   simulator tasks.
9. As a maintainer, I want task success or episode metrics to be the primary
   quality gate for simulator workloads, so that acceleration is judged by WAM
   task behavior rather than action-vector closeness alone.
10. As an agent, I want a machine-readable Evidence Bundle template, so that I
    can produce consistent review artifacts.
11. As a human reviewer, I want a short Evidence Summary, so that I can evaluate
    the claim without reading a large JSON object first.
12. As an open-source user, I want validation docs that do not depend on
    SuperPod, so that EazyWAM does not appear tied to a private cluster.
13. As a maintainer, I want private runtime details excluded from public docs,
    so that internal GPU operations do not become accidental product contract.
14. As a maintainer, I want serving acceleration validation recorded as a
    follow-up track, so that resident serving is not forgotten while the first
    runbook stays small.

## Implementation Decisions

- Add a public acceleration validation runbook centered on standard `wam`
  commands.
- Do not add SuperPod usage instructions, sbatch wrappers, account names,
  partitions, scratch paths, or recovery scripts to public repository docs.
- Use `wam run` for lightweight acceptance checks of profile selection, trace
  visibility, action contract, and latency smoke.
- Use `wam eval` for task-quality validation when the method affects simulator
  behavior or reported model quality.
- Use `wam compare` or method-specific report tools to summarize baseline vs
  variant evidence.
- Treat output drift as an audit signal by default. It becomes a hard gate only
  for methods that explicitly claim exact output preservation.
- Define Acceptance Validation as the first validation level. It proves a method
  is selectable, trace-visible, contract-compatible, and able to complete at
  least one relevant small workload without obvious task-quality failure.
- Define Measured Validation as the second validation level. It proves scoped
  latency or memory improvement under declared model, workload, runtime,
  fallback, task-quality, and performance conditions.
- Passing acceptance validation usually promotes a method to `experimental`,
  not `measured`.
- Only measured validation can promote a method to scoped `measured` status.
- Create two evidence layers:
  - Evidence Summary: short human-readable conclusion and caveats.
  - Evidence Bundle: machine-readable JSON plus command, trace, compare, metric,
    fallback, and privacy/redaction fields.
- First version should be documentation and templates only. Do not implement
  `wam evidence validate` yet.
- Use existing methods as examples for the first runbook:
  - `scheduler` as the measured scheduler-profile example.
  - `cuda_graph` as the exact-runtime measured example.
  - `teacache` as the experimental example where speedup-like signals do not
    justify a measured status.
- Record `wam serve` acceleration validation as a follow-up track for resident
  runtime, warmup, request latency, health, long-running stability, and batching.

## Evidence Template Decisions

The Evidence Summary should be concise and human-readable. It should include:

- status decision;
- scope;
- baseline;
- variant;
- task-quality result;
- performance result;
- profile applied/fallback status;
- output-drift audit result when available;
- caveats and non-generalization notes.

The Evidence Bundle should be structured and machine-readable. Version `0.1`
should include:

- schema version and evidence id;
- method/profile/family;
- status before and status decision;
- model id, workload, mode, seed, task scope, and trials;
- runtime context such as hardware class, accelerator count, dtype, runtime
  kind, and EazyWAM commit;
- baseline, variant, and compare commands;
- profile requested/applied/fallback status;
- acceptance validation result;
- measured validation result;
- output drift summary when comparable;
- artifact references or public-safe trace summaries;
- privacy and redaction notes.

The bundle must be public-safe. It should not contain private scheduler wrappers,
login instructions, account names, partitions, scratch paths, credentials, or
large raw cluster logs.

## Testing Decisions

- For the runbook itself, verify examples against the current public command
  surface and acceleration catalog.
- For Evidence Summary templates, verify that a maintainer can understand the
  claim without reading the JSON bundle.
- For Evidence Bundle examples, verify that required fields are present for
  acceptance and measured decisions.
- For `scheduler`, verify the template can express the existing measured
  evidence without adding new claims.
- For `cuda_graph`, verify the template can express exact-runtime evidence and
  fallback expectations.
- For `teacache`, verify the template can express an experimental result with
  blockers rather than a measured speedup.
- For docs-only PRD/runbook work, run `uv run ruff check .` and `git diff
  --check`.
- Do not require GPU, simulator, SuperPod, or private maintainer runtime access
  to complete the first runbook issue.

## Out of Scope

- Implementing `wam evidence validate`.
- Creating a formal JSON schema package.
- Automatically updating the acceleration method catalog from evidence bundles.
- Adding SuperPod usage docs to public git.
- Adding Slurm, PBS, LSF, or private scheduler wrappers.
- Re-running real GPU experiments as part of the first runbook issue.
- Making `wam serve --opt <method>` validation part of the first runbook's main
  path.
- Promoting any method to `measured` without existing supporting evidence.

## Further Notes

The follow-up issue split should keep the first version documentation-first.
Useful slices are:

1. Public acceleration validation runbook.
2. Evidence Summary and Evidence Bundle v0.1 templates.
3. Scheduler measured example.
4. CUDA Graph exact-runtime example.
5. TeaCache experimental example.
6. Future validator design.
7. Serving acceleration validation follow-up track.

The validator should be implemented only after the templates have been tested
against real method examples, because the schema will be easier to get right
after those examples reveal the necessary fields.
