# S-JEPA research team sources

These ten Markdown files preserve the original instructions from `.codex/agents/sjepa_*.toml` as canonical sources in the shared library. [roles.json](roles.json) records their descriptions and native names. The generator preserves the exact Codex instruction strings, including their confirmed ASTAD brief and study-specific working agreement. It does not apply the general research protocol as a replacement for the existing study protocol.

The [original team guide](../../../experiments/agents/sjepa-tutorials/README.md) remains the authority for the sixteen-task workflow, requirements and run templates. The general [shared library guide](../../README.md) explains maintenance, compatibility and invocation. Update both the relevant role source and study documentation when intentionally changing behavior, then regenerate and run both validators.

Codex keeps names such as `sjepa_literature`. Claude Code and OpenCode use `sjepa-literature`. The same conversion applies to all ten roles. The coordinator belongs in the main conversation, and workers receive bounded task assignments. OpenCode adapters add a host qualification because its tool and permission rules differ from Codex inheritance; Claude and OpenCode workers disable native recursive delegation.

This migration adds portable definitions. It does not execute tutorials, change the confirmed brief, or reinterpret an empty worldmodels brief as the ASTAD project.
