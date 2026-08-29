# Local work-product archive

The repository ignores `work/` because it contains generated artifacts,
checkpoints, cached poses, and presentation rendering products. On 2026-08-28,
the following stale local products were moved—without deletion—to
`work/archive/`:

- `gait-parity-pre-orbit-mask-2026-08-19/`: pre-repair GAVD smoke and real
  fixed-reflection runs. They predate the branch-specific orbit-closed mask
  contract and cannot support repaired-leakage claims.
- `presentation-build-2026-08-23/`: scratch assets, template inspections,
  render montages, and build scripts for superseded presentation work. The
  former `slides/` source directory is not included in this checkout.

The live `work/artifacts/` directory intentionally retains only the current
GAVD frozen Core11 probe, the StrokePIG frozen probe, and local pose artifacts
needed by the active runbook. The archive is recoverable local provenance, not
a Git-tracked source of truth.
