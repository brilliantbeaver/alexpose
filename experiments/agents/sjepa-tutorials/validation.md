# Validation of the saved research team

The team was created on November 15, 2026 in the project's America/Los_Angeles time zone, which was November 16 UTC. The user confirmed the populated ASTAD brief and requested separate storage for the team. These checks concern the agent definitions and coordination plan. Tutorial implementation, model training and scientific evaluation have not been launched.

## Checks that ran

The standard-library validator parsed all ten TOML profiles, checked their names and descriptions against the team registry, verified the confirmed brief's fingerprint and inspected all sixteen tasks for valid owners, independent review assignments, dependencies and acceptance criteria. It checked the four-agent concurrency limit and the single heavy-job limit. It also rejected seven deliberately invalid variants: duplicate roles, a nonexistent task owner, a dependency cycle, a nonexistent dependency, excessive concurrency, a task falsely marked accepted in the definition template, and a task assigned to its own author for review.

The command was run successfully from the repository root using the existing Python environment:

```bash
.venv/bin/python experiments/agents/sjepa-tutorials/validate_team.py --self-test
```

The installed `codex --version` reported `codex-cli 0.169.0`. A local, non-inference discovery check also completed successfully:

```bash
codex debug prompt-input 'Report the available sjepa agent names without executing any task.'
```

The resulting model-visible tool description listed all ten `sjepa_*` roles under its available-agent-types section. There were no diagnostic errors. The check establishes that this CLI discovers the profiles; it does not establish that every model, host or existing conversation reloads them, or that a spawned role has completed its future assignment. The raw debug output remains temporary because it also contains host context unrelated to these definitions.

The portable `--prompt` option prints the saved instructions for a host whose spawn tool accepts only a task message. That mode launches nothing and changes no configuration.

## Independent review

The requirements researcher mapped the confirmed brief to thirty requirements and checked the scientific and repository-specific constraints. A separate agent reviewed the role definitions, task graph, coordination protocol, requirements, templates and validator against the researched practices. That reviewer excluded `research.md` because it had authored that document; the coordinator inspected that research synthesis separately.

The review identified three issues. The coordinator inherited a worker instruction that prohibited delegation, so its definition now explicitly permits it to assign workers. T05 asked for a complete pipeline fixture before the model and evaluation code existed, so T05 now checks the environment and existing APIs, while a complete fixture is required after implementation and before T09 training. T15 integrated the final README and index after the principal review, so it now requires an independent check of those exact final artifacts before acceptance.

The reviewer rechecked all three corrections and independently reran the validator. All ten profiles, sixteen tasks and seven deliberate-invalid-input checks passed. No blocking findings remained in the reviewed files. Each task also has a separate review owner, and the validator rejects self-review. These checks concern the saved team's design; they do not establish future scientific outcomes.

## Practical limits

The task graph and file leases are instructions for the coordinator, not a background scheduler or filesystem access-control system. The actual host still controls permissions, model availability and concurrent execution capacity. Tutorial output location and run budget remain future T00 decisions. The ten profiles inherit model and permission choices rather than imposing new ones.

Existing settings, historical experiments, media files and unrelated working-tree changes were preserved. No API credentials, dependency environments or model settings were changed for this team setup.
