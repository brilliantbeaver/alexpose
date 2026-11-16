# Shared agents for AI/ML and agentic research

This directory is the source library for research agents that can be used by Codex, Claude Code, OpenCode and hosts that accept ordinary task prompts. It contains twelve general roles and the ten previously created S-JEPA roles. Each role has a focused purpose, concrete evidence requirements and boundaries for independent work. Saving the library does not start research jobs.

## How the library is organized

| Path | Purpose |
| --- | --- |
| [roles/](roles/) and [general-roles.json](general-roles.json) | These are the canonical general research instructions and their descriptions. |
| [protocol.md](protocol.md) | This is the shared working agreement, included in each generated general profile. |
| [teams/sjepa/](teams/sjepa/) | These sources preserve the existing S-JEPA specialist instructions and their original Codex names. |
| [templates/](templates/) | These templates describe assignments, handoffs, reviews and run records. |
| [research.md](research.md) | This review explains the primary sources, compatibility findings and design choices. |
| [scripts/sync_agents.py](scripts/sync_agents.py) | This standard-library generator creates and checks the native profiles. |
| [generated.json](generated.json) | This inventory records generated paths and content fingerprints for detecting drift. |
| [validation.md](validation.md) | This report distinguishes checks that actually ran from unverified runtime behavior. |

The `.agents/roles/` path is our repository convention. It is not a universal subagent discovery standard. Each supported tool receives native profiles in its documented directory. The generator includes the complete canonical prompt in each profile, so a worker does not depend on its parent's conversation to learn its responsibilities. It preserves file references to project instructions and task evidence, which must remain available in this repository.

| Host | Generated project files | How to invoke them |
| --- | --- | --- |
| Codex | [../.codex/agents/](../.codex/agents/) contains TOML definitions. | Ask the main agent to use a named specialist. If its spawn tool does not support named profiles, supply the full prompt instead. |
| Claude Code | [../.claude/agents/](../.claude/agents/) contains Markdown with YAML metadata. | Ask Claude to use a named specialist, or run `claude --agent research-coordinator` for a coordinator session. |
| OpenCode | [../.opencode/agents/](../.opencode/agents/) contains Markdown targeting the documented V2 format. | Select the primary `research-coordinator` agent or ask the primary agent to use a named worker. |

The [Codex](https://learn.chatgpt.com/docs/agent-configuration/subagents), [Claude Code](https://code.claude.com/docs/en/sub-agents) and [OpenCode](https://opencode.ai/docs/agents) documentation explains native discovery and invocation. Start a fresh session after creating a native agents directory. Actual capabilities depend on the installed host and version. OpenCode is not installed in the environment used to create this library; its adapters have not been tested in that runtime.

## Choose a focused role

| General role | Use it when you need this result. |
| --- | --- |
| `research-coordinator` | A coherent plan, assignments, resources, decisions and accepted evidence. |
| `research-literature` | Verified primary sources, method distinctions and testable hypotheses. |
| `research-data` | Data provenance, eligibility, missingness, related sources and a cohort contract. |
| `research-representation` | Explicit raw-data-to-feature, token or embedding mappings, units and shapes. |
| `research-model` | A faithful model implementation, controlled variants and traceable training runs. |
| `research-evaluation` | Frozen endpoints, isolated partitions, fair comparisons and uncertainty. |
| `research-agent-systems` | Tool, memory and delegation designs evaluated against isolated tasks and baselines. |
| `research-reproducibility` | Independent replay with declared inputs, environment and cache state. |
| `research-software` | Shared APIs, robust adapters and execution machinery with integration evidence. |
| `research-tutorials` | Progressive teaching materials grounded in runnable examples and accepted claims. |
| `research-visuals` | Accurate editable diagrams and scientific figures with rendered inspection. |
| `research-reviewer` | Independent findings against exact artifacts, claims and acceptance criteria. |

Use only the roles needed for an assignment. The main conversation should coordinate the work; avoid launching a second coordinator as a worker. The twelve roles do not require twelve simultaneous agents. The protocol offers three workers and one heavy computation as conservative initial limits, subject to the actual host, machine and task. Those numbers are local choices, not universal performance recommendations.

## Start an authorized project

From the repository root, validate the library before starting the chosen host:

```bash
.venv/bin/python .agents/scripts/sync_agents.py
```

Then supply a concrete prompt such as:

```text
Use the shared team in .agents/README.md and act as research-coordinator.
Execute the authorized brief at <actual populated brief path>.
Read the brief and relevant project instructions before assigning work.
Create a run record in the chosen project's output directory from
.agents/templates/run.json. Explicitly delegate bounded assignments to
appropriate research specialists, with disjoint writable paths and
independent acceptance checks. Continue through implementation and
verification within the authorized scope. Report actual artifacts,
checks, remaining uncertainty and each worker's status.
```

Replace the placeholder with the actual brief. Check that the file has content; the worldmodels brief was empty when this library was created. Definitions do not fill an empty brief or silently substitute the ASTAD study. When a material scientific scope decision is missing, the coordinator records the gap and advances independent work while obtaining clarification.

Codex CLI users can inspect worker threads with `/subagents`. Other hosts expose their own activity or child-session views. Ask the coordinator for the assignment, progress, blockers and artifacts of each worker; the persistent run record provides continuity across sessions. There is no background scheduler or shared dashboard implemented by this folder.

For a host without native role selection, print a complete portable prompt:

```bash
.venv/bin/python .agents/scripts/sync_agents.py --prompt research-agent-systems
```

Pass that text and a completed task assignment to the host's delegation tool. If no delegation tool exists, execute tasks sequentially and report that separate-agent review has not occurred.

## Maintain one source of truth

Edit a general role under `roles/` and its metadata in `general-roles.json`. Edit the common agreement in `protocol.md`. For S-JEPA, edit the corresponding canonical team source and update its study documentation when behavior changes. Then regenerate and check:

```bash
.venv/bin/python .agents/scripts/sync_agents.py --write
.venv/bin/python .agents/scripts/sync_agents.py
.venv/bin/python .agents/scripts/test_sync_agents.py
.venv/bin/python experiments/agents/sjepa-tutorials/validate_team.py --self-test
```

The generator refuses to overwrite unmanaged or locally edited native files. Review a conflict, preserve any intended changes in the canonical source, and restore the conflicting output to its recorded generated content before retrying. Retired generated profiles require explicit removal and inventory reconciliation. Do not resolve conflicts by blindly deleting the inventory. A content fingerprint detects drift; it is not a cryptographic trust or approval mechanism.

Generated copies are intentional format adapters, not independent editable definitions. Symlinks would only change paths and would not translate TOML into Markdown metadata. No `.agents/skills/` wrappers were created because these definitions describe specialist workers. A future reusable procedure can be packaged as a skill separately.

## Permissions and review

The adapters do not choose a model, provider, credentials or approval bypass. Native workers disable the host's named delegation tool where supported: Claude uses `disallowedTools: Agent`, and OpenCode V2 denies the `subagent` action. Codex worker delegation is constrained by instructions in these profiles. Other tools, shell commands or connectors may still expose capabilities, so these limits do not constitute a sandbox or prevent every possible form of delegation.

OpenCode workers use the host's permission configuration with the added delegation denial; they do not claim to inherit Codex's permission semantics. Reviewers avoid implementation writes by instruction and use assigned scratch space for checks. Neither a file lease nor a role name establishes operating-system isolation or independent evidence.

The [existing S-JEPA team](../experiments/agents/sjepa-tutorials/README.md) retains its confirmed ASTAD brief, sixteen-task workflow and historical preservation rules. Its ten Codex IDs retain underscores for compatibility, while Claude and OpenCode names use hyphens, such as `sjepa_literature` versus `sjepa-literature`. General research IDs use hyphens in all three hosts. Do not combine the two coordinator workflows or treat study-specific assumptions as universal research defaults.
