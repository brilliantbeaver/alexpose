# Measured gait-prediction study

The active 43-slide deck presents completed real-data experiments. Its central result is that useful normal prediction does not guarantee better anomaly detection. It reports negative fusion results, the failed sparse timing-support gate, the actual JEPA and Random Forest comparisons, and the independent participant replication.

Slides 41–43 add a clearly labeled post-outcome appendix for [draft03](../docs/draft03/README.md). The positive completion interpretation uses frozen predictions: a 19.5% GAVD distal-task benefit, a 5.2% laboratory benefit, and a separate 7.7% anatomical-mask advantage on the distal task. The unchanged uniform task remains uncertain. The [paired audit](../docs/draft03/advisory/completion_audit_v1/README.md) records exact comparators, conditional intervals and the original negative primary findings. No model or score was changed for this appendix.

Open the [offline viewer](index.html), [vector PDF](tutorial.pdf), or [detailed slide notes](tutorial.md). The [planning deck](archive/iteration02_plan/tutorial.pdf) and [draft01 deck](archive/draft01_completed/tutorial.pdf) are immutable historical archives.

The GAVD evidence is run `d66e1b126c9d9a9adf1b`. The independent laboratory evidence is `external_fb9d0d23b56ce29fd846`. [Input hashes](result_input.json), [build metadata](build_manifest.json), and [render checks](review/validation.json) preserve the evidence trail. Interpretive diagrams are illustrations; tables and plots use saved real outputs.

Rebuild from the repository root:

```bash
uv run --project experiments/aaai python experiments/aaai/scripts/build_iteration02_result_slides.py --run-dir experiments/aaai/artifacts/iteration02/d66e1b126c9d9a9adf1b --external-dir experiments/aaai/artifacts/iteration02/external_fb9d0d23b56ce29fd846
```

The builder requires a completed verified run, exports editable SVG/PDF and PNG previews, checks links and bounds, and renders every final PDF page for inspection. See [the independent tutorials](../12_observation_population_and_video_context.ipynb) and [draft02 readiness](../docs/draft02/README.md).
