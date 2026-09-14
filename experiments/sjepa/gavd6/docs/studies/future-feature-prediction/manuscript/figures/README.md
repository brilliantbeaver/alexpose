# Figure provenance

Figures 01–05 are editable vector diagrams. Coordinates and trajectories are schematic, not extracted poses. Figure 01 uses reported laterality and direct-v3 aggregates while keeping their metrics separate. SVG files are editable vectors; PDF companions support manuscript typesetting.

Figure 06 displays paired source-bootstrap intervals from the cached panel, distinguishing the primary posture contrast from secondary comparisons. Its original input was `outputs/iclr-bridge-cached-20260911/reports/panel-report.json`. That run is absent from this checkout; the [preserved report copy](../../accessibility/evidence/cached-panel-report.json) and its [provenance record](../../accessibility/evidence/README.md) remain available for inspection.

From the repository root, regenerate Figures 01–05 with:

```bash
.venv/bin/python scripts/research_directions/target_accessibility/build_figures.py
```

The command also regenerates Figure 06 if the original report is installed. Otherwise it explicitly reports that Figure 06 was skipped and keeps its existing SVG/PDF pair. A successful command therefore does not imply that Figure 06 was rebuilt or that the original run was reverified.
