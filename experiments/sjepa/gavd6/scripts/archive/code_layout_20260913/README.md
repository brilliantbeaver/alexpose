# Source before the September 2026 code refactor

[source.tar.gz](source.tar.gz) contains **287 unchanged files at their original paths**: Python implementations and tests, script/Slurm entry points, calibration resources, dependency files and five numerical protocols. [manifest.json](manifest.json) records each digest and the three original software fingerprints. This is a numerical-replay source snapshot, not a dataset or a result bundle.

Keep this directory outside automatic path/name rewrites. Its filenames inside the archive must remain historical. The code-organization review verified every digest and reproduced the three fingerprints in a separate process that imported only the snapshot's `gavd6_sjepa` modules.

## Use the correct implementation

Renamed classes have different Python import paths. Old fitted Joblib objects can require the original modules even when the equations are unchanged. Extract this snapshot into a separate directory and put its `src` first on the interpreter search path before importing the package. Run with the dependencies specified by the archived `uv.lock`; supply original data and run locations explicitly. Confirm `gavd6_sjepa.__file__` and the loaded study modules resolve inside the extracted snapshot. An editable installation of the current checkout must not shadow it.

For example, from the current project root, inspect the old CLI without running an experiment:

```bash
mkdir -p /tmp/gavd6-code-before
tar -xzf scripts/archive/code_layout_20260913/source.tar.gz -C /tmp/gavd6-code-before
.venv/bin/python -I -c 'import sys, runpy; from pathlib import Path; root=Path("/tmp/gavd6-code-before"); sys.path.insert(0,str(root/"src")); import gavd6_sjepa; assert Path(gavd6_sjepa.__file__).is_relative_to(root); sys.argv=["gavd6","--help"]; runpy.run_module("gavd6_sjepa.command_line_interface",run_name="__main__")'
```

The scaling and accessibility readers enforce their strict software identity contracts. The prediction gate retains its preexisting behavior of recording changed code/runtime provenance while validating frozen configuration and artifacts. This refactor does not change either rule. Prefer the snapshot when reconstructing historical fitted models; do not modify saved receipts to make a mismatch disappear.

The full local recovery archive, including source notebooks and their saved outputs, is at `work/code-organization/20260913/replay-source-before.tar.gz`. The compact archive here excludes those large notebook payloads. Historical datasets, checkpoints and remote result bundles must still be supplied separately. No archived experiment is executed merely by extracting this archive.
