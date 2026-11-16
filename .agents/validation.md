# Validation of the shared research agents

These checks were performed on November 15, 2026 in America/Los_Angeles, which was November 16 UTC. They establish properties of the saved definitions and adapters. They do not establish successful scientific experiments or a measured benefit from using multiple agents.

## Checks that ran

The library contains twelve general research roles and ten preserved S-JEPA roles. The standard-library generator produced 66 native profiles across Codex, Claude Code and OpenCode. Its subsequent default check found no drift. The existing S-JEPA validator still accepted all ten profiles and sixteen tasks and rejected its seven deliberate invalid cases. The S-JEPA Codex instruction strings were preserved while their canonical authoring location moved into this library.

Nine isolated regression tests passed. They exercise native prompt translation and worker modes, deterministic regeneration, source-change propagation, unmanaged-file collisions before writes, preservation of local edits, rejection of symlink ancestors, rejection of a symlinked generated inventory, duplicate identities and escaping sources, preservation of legacy S-JEPA IDs and semantics, and refusal to silently remove retired profiles. Some tests exercise more than one related boundary.

The commands were:

```bash
.venv/bin/python -B .agents/scripts/test_sync_agents.py
.venv/bin/python -B .agents/scripts/sync_agents.py --write
.venv/bin/python -B .agents/scripts/sync_agents.py
.venv/bin/python -B experiments/agents/sjepa-tutorials/validate_team.py --self-test
```

All 22 Codex TOML files were parsed with the standard library. All 44 Markdown metadata sections were parsed using the existing environment's PyYAML, with checks on names and native fields. PyYAML is not required by the generator.

Codex CLI 0.169.0 completed a local `debug prompt-input` discovery check without inference and with no diagnostic errors. Its model-visible available-role tool description included all 22 role IDs. Raw debug output remains temporary because it includes unrelated host context. This is evidence of local profile discovery, not a completed native worker invocation.

Claude Code 2.1.186 was installed, and its help advertised `--agent`. Native profile parsing was checked locally; a Claude model session and native discovery were not tested. OpenCode was not installed. Its definitions target the current documented V2 metadata and were parsed locally, but discovery, permission behavior and execution in OpenCode remain unverified.

## Independent review and corrections

The compatibility researcher wrote `research.md` and independently reviewed the generator and shared protocol. It reproduced a concrete flaw in which a symlinked output ancestor could redirect a write outside the intended repository. The generator now rejects symlinked destinations, ancestors and inventory files before writing, and regression tests cover those cases. The reviewer also identified that S-JEPA's inherited-permission language needed qualification for OpenCode; the generated OpenCode prompt now states that the host's actual controls are authoritative.

The general-role author independently reviewed the research synthesis, guide, protocol and generator, excluding the roles it authored. Its review found no blocking issues in that scope. The coordinator separately checked the role definitions, source mappings, generated metadata and agreement with the existing study.

The compatibility researcher then independently rechecked both corrections, all twelve general role bodies, the protocol, native Claude and OpenCode samples and the integrated guides, excluding its own research synthesis. It reran all nine regression tests and the drift check successfully and found no remaining blocking issues. A separate link check verified that all 38 local Markdown targets in the shared documentation and original team guide existed.

## Practical limits

The generator checks the formats and ownership rules implemented here; it is not a full replacement for each product's evolving schema or runtime. File leases and reviewer independence still require coordinated behavior. Native delegation restrictions do not prevent every possible action through shell tools or connectors. Models, provider availability, approval policies and access controls remain host concerns.

No scientific study was launched, no global settings were changed, and no provider credentials or model selection were added. The library uses generated native copies rather than symlinks because each host needs different metadata and sometimes different names.
