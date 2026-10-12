# Reproduce the manuscript artifacts

These commands analyze existing evidence and build documents; they do not launch training or change source experiment files. Run from the repository root with its Python environment. The LaTeX command below writes to a scratch directory, preserving the retained PDFs and logs. Figure builders regenerate shared assets in place; copy the repository to a scratch location first if regenerating figures must leave the retained package untouched.

```sh
.venv/bin/python docs/iclr/evidence/independent_evidence_audit.py
.venv/bin/python docs/iclr/scripts/build_figures_v04.py
.venv/bin/python docs/iclr/scripts/build_laterality_figure.py
mkdir -p /tmp/iclr-v07-rebuild
(cd docs/iclr && tectonic versions/v07/paper-v07-final.tex -Z search-path=. --outdir /tmp/iclr-v07-rebuild --keep-logs)
.venv/bin/python docs/iclr/scripts/validate_artifacts.py
```

Sources are grouped under `versions/vNN/`; shared resources remain at `docs/iclr`. The `-Z search-path=.` option resolves the original figure, template, and bibliography paths without editing the hash-bound manuscript sources. Run the command from `docs/iclr` as shown. Substitute the version and omit `-final` to rebuild a review-stage draft. Retained historical compiler logs are in `versions/vNN/logs/`; fresh logs go to the scratch output directory. The validator checks the retained deliverables and logs, not the scratch rebuild.

Figure scripts read CSV/JSON evidence under `outputs/iclr` and record input SHA-256 hashes. `build_figures_v04.py` supplies the final conceptual, architecture, objective-tradeoff and failure/interval figures. The separate naming builder supplies the final laterality figure. Earlier version-specific builders remain for historical figures. The discarded main-text repair and probe panels remain available in the supporting evidence; their omission is editorial, not an erasure of results.

The exact executed operators, calibration reductions and updates are documented in [reproducibility-methods.md](evidence/reproducibility-methods.md). The [claim ledger](CLAIM-LEDGER.md) maps reported measurements to exported files. The [independent audit](reviews/evidence-initial.md) explains hierarchical weighting, missingness and inferential units.

The local compact packet supports recalculating summaries and plots. It does not contain all source motion assets, renderer inputs, poses, checkpoints or trajectories needed for end-to-end rerunning. Source experiment launch instructions remain in `slurm/gait-fidelity`; executing those is separate work and was not done during manuscript preparation. Tutorial fixtures are excluded from the paper.

The official unmodified ICLR 2027 style files are retained under `template/iclr2027`. Tectonic may download TeX dependencies on its first run. The final PDFs have nine main-text pages; mandatory AI disclosure and references begin on page10. The user-specific research audit files contain local paths and are working records, not a claim that an anonymous public code release has been prepared.
