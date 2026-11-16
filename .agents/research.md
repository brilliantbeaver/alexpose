# Research behind the shared agent library

This review was checked on November 15, 2026 in America/Los_Angeles, which was November 16 UTC. It distinguishes documented product behavior from choices made for this repository. Product documentation can change, so compatibility should be rechecked when upgrading a coding client. Saving a role does not demonstrate that it has run or that its scientific conclusions are valid.

## What a shared directory can provide

The repository uses `.agents/` as the common home for role instructions, shared working agreements, templates and team descriptions. This is an authoring convention we control. The checked documentation does not establish `.agents/roles/` as a universal location that coding tools automatically discover.

The Agent Skills specification defines a different artifact: a directory containing `SKILL.md`, with optional scripts, references and assets. It provides a portable format for reusable capabilities. It does not define a universal subagent registration format. A research role may use skills, but placing its instructions under `.agents/` does not turn it into a registered subagent. [Agent Skills specification](https://agentskills.io/specification).

## Native discovery and format differences

| Client | Project discovery location | Native definition | Consequence for this library |
| --- | --- | --- | --- |
| Codex | `.codex/agents/` | One TOML file per role, with `name`, `description` and `developer_instructions`. | Generate TOML adapters from the canonical instructions and preserve existing S-JEPA identifiers. |
| Claude Code | `.claude/agents/` | Markdown with YAML metadata containing `name` and `description`. | Generate a separate Markdown adapter with a complete prompt and compatible identifier. |
| OpenCode V2 | `.opencode/agents/` | Markdown with YAML metadata; the filename supplies the agent identifier. | Generate native V2 metadata and set each worker's mode explicitly. |

Codex documents project and personal custom-agent directories and inherited permissions, provider and MCP configuration. A custom agent file cannot replace those inherited settings. This library therefore leaves model and provider selection to the running session and does not introduce permission bypasses. Definition discovery remains distinct from a successful invocation. [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Claude Code loads project agents from its own directories and gives each worker a separate prompt and task context. Names and descriptions guide selection, while optional tool restrictions control capabilities. Current documentation allows nested spawning by default; the older claim that subagents can never delegate is inaccurate for current releases. Our workers prohibit recursive delegation as a local design choice. The main conversation can adopt the coordinator role, including through `claude --agent` where supported, and assign workers directly. A newly created agent directory can require restarting Claude Code before discovery. [Claude Code subagents](https://code.claude.com/docs/en/sub-agents).

OpenCode derives an agent's identity from its file path, and nested directories become part of that identity. Its custom-agent default is primary mode, so a worker needs `mode: subagent`; a coordinator intended for either context can use `mode: all`. A worker receives fresh context. Its permissions come from its configured rules, so Codex's permission inheritance must not be assumed to apply unchanged. Native V2 rules use an ordered `permissions` list, with `action`, `resource` and `effect`. Later matching rules take precedence. [OpenCode agents](https://opencode.ai/docs/agents).

OpenCode's migration guide is especially relevant. V2 still translates supported V1 agent definitions, but a file using a V2 marker such as `permissions`, `request`, `system` or `disabled` is interpreted as V2 throughout. Legacy fields inside it can be ignored with warnings. New V2 adapters must avoid mixing in V1 `name`, `tools` or singular `permission` fields. The guide recommends `.opencode/agents/` and requires checking behavior before relying on migrated configuration. This library targets native V2 rather than claiming compatibility with every historical release. [OpenCode migration guide](https://opencode.ai/docs/migrate-v1#agent-files).

## Why the library generates adapters

Canonical Markdown and a manifest give reviewers one place to maintain responsibilities and identities. Generated adapters translate that content into each host's format. A symbolic link can share bytes, but it cannot translate TOML into YAML metadata or reconcile different permission rules. Generation also provides a concrete way to check for drift without maintaining three independent copies by hand.

The generic `research-*` roles and the existing S-JEPA team serve different scopes. General roles support future AI/ML and agentic research. S-JEPA roles retain their confirmed brief, scientific distinctions and study-specific dependencies. Sharing a directory does not authorize changing a frozen experiment or applying its quality thresholds to unrelated work. The shared working contract belongs in [protocol.md](protocol.md); task packets supply the current brief, output location and accepted inputs.

## Coordination practices supported by research

Anthropic recommends beginning with simple, composable workflows and adding autonomy where it helps. Its discussion includes a coordinator that delegates and an evaluator that checks outcomes. The article explicitly warns that its original tooling discussion has aged. We use those architectural ideas while checking current product documentation for file formats and commands. A larger roster is not evidence of better performance. [Building effective agents](https://www.anthropic.com/research/building-effective-agents).

Anthropic's research-system account reports that vague assignments caused duplicated work and missing coverage. It recommends clear objectives, output formats, tool guidance and boundaries, and emphasizes the resource cost of additional agents. We apply that by giving every assignment a concrete result, acceptance evidence and a stopping condition. Independent literature, data and API investigations can run together; changes to the same files require coordinated ownership. Performance reported for Anthropic's system does not establish performance for ours. [How we built our multi-agent research system](https://www.anthropic.com/research/multi-agent-research-system).

Context-engineering guidance supports focused worker context and concise summaries, with durable notes preserving state across sessions. Our handoffs therefore identify artifacts, commands, findings and unresolved questions rather than repeating entire logs. The coordinator must retain the evidence paths needed to inspect a claim. A condensed report cannot substitute for an unavailable source or silently discard a methodological limitation. [Effective context engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents).

## Scientific acceptance and practical limits

Agent workflow evaluation needs its own evidence. Anthropic recommends checking both an agent's trajectory and the resulting state, using explicit tasks and graders, and accounting for variation across repeated trials. This motivates the `research-agent-systems` role: keep development and held-out tasks separate, record retries and tool state, compare with a simple baseline, and report success together with measured resource use. We have defined these checks as future acceptance criteria; this library has not demonstrated an improvement in agent performance. [Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents).

The NeurIPS checklist asks researchers to make claims, assumptions, experimental settings, reproducibility routes and limitations explicit. We use it as a research-quality reference, without assuming another venue has identical rules. Research handoffs should record data and split identities, environments, configurations, seeds, commands and outputs. Planned experiments, synthetic checks and measured findings must remain distinguishable. A successful implementation check does not establish generalization or clinical validity. [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist).

File ownership, resource budgets and a prohibition on worker delegation are local coordination policies. Prompt text is not an operating-system boundary. Tool restrictions also differ by host, and shell access can permit writes even when dedicated edit tools are restricted. The library must describe the controls actually configured and avoid calling a role technically read-only on the strength of prose alone. Validation should check generated content and report which clients were available for discovery checks; actual execution and scientific outcomes require separate evidence.
