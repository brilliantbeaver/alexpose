# Research behind the S-JEPA tutorial agent team

This review explains how published guidance shaped the local team definitions. The sources were read on November 15, 2026, in the project's America/Los_Angeles time zone. They support practical design choices, but they do not demonstrate that this particular team will improve tutorial quality or research results. That requires evaluation after the team is used.

The team has ten defined roles: coordinator, literature researcher, data steward, pose engineer, model engineer, evaluation scientist, tutorial author, visual designer, independent reviewer, and reproducibility engineer. These are responsibilities that can be assigned when needed. They are not ten agents that should run continuously.

## What the primary sources establish

### Begin with a simple coordination structure

Anthropic describes a coordinator that divides unfamiliar work into focused assignments and combines the results. It also describes a separate evaluator that provides feedback against clear criteria. Its guidance favors increasing complexity only when there is evidence that the added structure helps. The article now notes that some of its tooling discussion has aged, so its architecture principles are more useful here than its product examples. [Building effective agents, December 19, 2024](https://www.anthropic.com/research/building-effective-agents).

For this project, the coordinator owns the plan and accepts completed work. Specialist roles receive bounded assignments, and an independent reviewer checks important scientific and implementation claims. Routine edits stay with their existing owner rather than creating another agent solely to make a small change. This is our application of the pattern, not a performance result from the article.

### Give each assignment a clear boundary and a concrete result

Anthropic's account of its research system says that a worker needs an objective, an expected output, suitable tools and sources, and clear boundaries. It reports that vague delegation led to duplicated searches and gaps. It also cautions that research often has more independently parallel work than coding, and that additional agents consume substantial resources. Its reported performance improvements came from its own research evaluations and should not be transferred to this repository. [How we built our multi-agent research system, June 13, 2025](https://www.anthropic.com/research/multi-agent-research-system).

Each local assignment therefore names the question or change, required inputs, owned files, expected artifacts, acceptance checks, and dependencies. Literature review and dataset auditing can proceed independently. Two agents editing the same notebook generator or environment file must work in sequence. We do not treat the number of agents as a measure of research depth.

### Keep the handoff concise and preserve the evidence elsewhere

Anthropic recommends focused context, structured notes that survive context changes, and specialist work that returns a compact summary to the coordinator. It warns that too much unrelated material can obscure useful information and that aggressive summarization can discard important decisions. This supports keeping evidence available without copying every search result or execution log into every conversation. [Effective context engineering for AI agents, September 29, 2025](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents).

The local handoff should contain the outcome, files changed, checks actually run, remaining uncertainty, and the next dependency. Long logs and detailed research notes belong in named files that the coordinator or reviewer can inspect. A resumed agent reads the latest task record and relevant evidence rather than assuming that a previous conversation is still available. Compact notes must retain rejected hypotheses when those decisions affect the method.

### Make progress observable and test completion claims

Anthropic's long-running-agent study describes failures in which an agent attempted too much at once, left unclear partial work, or declared completion without testing the complete behavior. It recommends incremental work, explicit requirements, and persistent progress records that let the next session recover the state of the project. Its examples concern application development rather than scientific validity. [Effective harnesses for long-running agents, November 26, 2025](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents).

For the tutorials, a requirement is complete only when its stated evidence exists. A notebook that parses is not equivalent to one that runs from a fresh kernel. A successful small demonstration does not establish a full training result. Each handoff records which of these checks ran and which remain open. The team preserves a usable checkpoint when a run stops; it does not invent a result to make the plan look complete.

### Treat shared state and failed handoffs as engineering problems

Microsoft's architecture guidance calls for explicit state management, timeouts, bounded retries, error reporting, and validation of one agent's output before another consumes it. It identifies concurrent changes to shared mutable state and ignored resource constraints as common mistakes. It also recommends combining sequential and parallel stages when their dependencies differ. The guidance is platform-neutral even though its implementation examples include Microsoft products. [AI agent orchestration patterns, updated February 12, 2026](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns).

The coordinator therefore maintains the shared task ledger and assigns one writer to each file or directory at a time. Workers submit changes through their own handoffs rather than competing to rewrite the ledger. An invalid report, failed dependency, or missing input keeps dependent work pending. A failed attempt gets a bounded diagnosis and a changed recovery plan; repeating the same unsuccessful command indefinitely is not recovery.

### Use independent checks and distinguish instructions from enforcement

The current Claude Code documentation distinguishes model-followed instructions from controls enforced by the surrounding software. It recommends executable completion checks, separate workspaces for concurrent edits, and a fresh reviewer who has not followed the author's reasoning. It also explains that a new worktree does not automatically contain ignored dependencies and local configuration. This review uses the maintained documentation rather than the older, explicitly unmaintained best-practices blog. [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices).

Our role descriptions and file-ownership rules are instructions, not filesystem security boundaries. Acceptance requires recorded checks and review of the actual files. The reviewer receives the brief, artifact paths, and test evidence without being asked to endorse the author's conclusion. If separate worktrees are needed, the coordinator accounts for local data and environment prerequisites. This document does not change the project's permissions or claim that another product's commands work in Codex.

### Make scientific evidence reproducible

The official NeurIPS checklist frames reproducibility and transparency as research responsibilities and asks authors to explain what is and is not supported. Its purpose is broader than coding correctness. We use it as a research-quality reference, not as an assertion that the ASTAD workshop uses the same submission requirements. The actual target venue and current deadlines must be checked separately. [NeurIPS Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist).

The reproducibility role records the data identity, split definitions, software environment, configuration, seeds, commands, and produced artifacts. The evaluation role connects each numerical claim to a recorded run and describes important limitations. Tutorial examples, proposed experiments, and executed findings remain distinguishable. Acceptance must not depend on obtaining a favorable scientific outcome.

## Local choices and why they fit this project

The following decisions are project-specific applications of the evidence above. Their exact limits are not universal recommendations from the cited sources.

| Local choice | Reason and practical consequence |
| --- | --- |
| The coordinator can use up to three workers at once. | This session exposes four concurrent agent slots in total. The coordinator occupies one slot and must check the available capacity again in future sessions. A role can be reused in a later wave instead of keeping all ten roles active. |
| Only one substantial training or pose-extraction job runs at a time by default. | The tutorials must work on a laptop. This avoids competing jobs exhausting memory or obscuring timing measurements. The coordinator can revisit the limit after measuring the actual machine and workload. |
| Every write assignment has a named owner. | Shared notebook generators, configuration files, manifests, and documentation indexes are common collision points. A worker that needs another owner's file reports the dependency and lets the coordinator transfer ownership or schedule the change. |
| Literature, data, and pose investigations begin before model claims are fixed. | The method must match the papers, the available annotations and videos, and the shared Ambient interfaces. Missing evidence becomes an explicit decision or limitation rather than a hidden assumption. |
| Model and Random Forest work use the same declared evaluation protocol. | Source-grouped splits, training-only preprocessing, and untouched evaluation data make comparisons interpretable. These are local scientific controls, not claims established by agent-orchestration articles. |
| Notebook and visual work follow accepted interfaces. | Tutorial authors can draft the explanation early, but runnable examples and diagrams must reflect the implemented data flow. Generated notebooks and graphics are updated through their source generators. |
| Review and reproducibility are separate assignments. | The reviewer challenges method and code claims. The reproducibility engineer checks whether recorded instructions reproduce the stated behavior. Neither role substitutes for the other's evidence. |
| The coordinator records a bounded task before dispatch. | The task states the deliverable, expected verification, dependencies, and a stopping condition. Resource usage is reported when available. Numeric token or monetary budgets are not invented when the user has not supplied them. |
| Stable evidence survives a change of agent or session. | A task record points to source documents, implementation files, checks, and unresolved issues. Before resuming, the coordinator checks whether those inputs changed and whether previous results are still applicable. |

## How to assess the team after it is used

The first trial should be a small, complete tutorial slice, such as reading one eligible sequence, passing it through the declared pose representation, and running a short synthetic model check. The coordinator should record which requirements passed, whether agents duplicated work, whether file ownership was respected, and how much repair was needed after review. Successful delegation means the handoff makes the next step easier to verify; it does not merely produce more files.

Test the workflow with concrete failure cases as well. A missing paper must remain an unresolved citation. A video that cannot cover the requested frames must not become a fabricated complete sample. A worker's claim that a notebook ran must be supported by the recorded command and output. A reviewer must be able to reject a result whose data split leaks a source across training and evaluation. These cases exercise the tutorial team's actual risks without requiring a large training run.

The local role definitions are durable instructions for future use. Persisting them does not mean that all roles have been launched, that a particular client has loaded them automatically, or that the tutorial experiments have been executed. Agent discovery and configuration compatibility require separate verification against the installed runtime. The coordinator must use the task-prompt fallback when that runtime does not expose named role selection.
