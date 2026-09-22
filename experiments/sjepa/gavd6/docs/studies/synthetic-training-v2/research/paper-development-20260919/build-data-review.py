"""Build a local review page from retained diagnostics without altering results.

The page links to existing images; keep it inside this checkout. It does not
contain RGB clips, infer annotations, or record a human review automatically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/full-updates-2000-seed-17/updates-2000-seed-17"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    import os
    return Path(os.path.relpath(path, OUTPUT)).as_posix()


def build():
    selection = json.loads((SOURCE / "selected-windows.json").read_text())
    rows = []
    for person in selection["selection"]:
        name = person["canonical_person_id"]
        short = name.split("::")[-1]
        window = person["window_id"]
        for extractor in ("hrnet_w32", "rtmpose_m", "vitpose_base"):
            for condition in ("clean", "blur", "obstruction", "blur_obstruction"):
                prefix = f"trajectory-BioMotionLab_NTroje-{short}-{window}-{extractor}-{condition}-"
                matches = [p for p in (SOURCE / "images").glob(prefix + "*.png")
                           if re.fullmatch(re.escape(prefix) + r"[0-9a-f]{10}\.png", p.name)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one retained figure for {prefix}: {matches}")
                path = matches[0]
                rows.append(dict(person=name, short=short, window=window,
                                 extractor=extractor, condition=condition,
                                 image=relative(path), sha256=sha(path)))
    training = []
    for path in sorted((SOURCE / "images").glob("training-*.png")):
        match = re.fullmatch(r"training-(.+)-17-[0-9a-f]{10}\.png", path.name)
        if not match:
            raise ValueError(f"Unexpected training figure: {path}")
        training.append(dict(method=match[1], image=relative(path), sha256=sha(path)))
    summaries = [dict(name=name, image=relative(OUTPUT.parents[1] / "results/seed17-complete-analysis-20260919" / filename),
                      sha256=sha(OUTPUT.parents[1] / "results/seed17-complete-analysis-20260919" / filename))
                 for name, filename in (("Paired effects across eight people", "paired-effects.png"),
                                        ("Exploratory ankle-separation amplitude", "amplitude-diagnostics.png"))]
    if len(rows) != 48 or len(training) != 8:
        raise ValueError("This page expects the verified seed-17 diagnostic package")
    for row in rows + training + summaries:
        if not (OUTPUT / row["image"]).is_file():
            raise FileNotFoundError(row["image"])
    payload = dict(trajectories=rows, training=training, summaries=summaries,
                   selection_policy=selection["policy"], seed=17,
                   source=str(SOURCE.relative_to(ROOT)))
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    document = (OUTPUT / "data-review-template.html").read_text().replace("__DATA__", encoded)
    (OUTPUT / "data-review.html").write_text(document)
    receipt = dict(source=str(SOURCE.relative_to(ROOT)), generator_sha256=sha(Path(__file__)),
                   html_sha256=sha(OUTPUT / "data-review.html"),
                   selected_windows_sha256=sha(SOURCE / "selected-windows.json"),
                   trajectory_figures=len(rows), training_figures=len(training), summary_figures=len(summaries),
                   displayed_people=4, evaluated_people=8, displayed_physical_windows=4,
                   evaluated_physical_windows=32, seed=17,
                   selection_policy=selection["policy"],
                   limitations="Retained diagnostic plots only; RGB clips, raw neural arrays and human quality decisions are not supplied.",
                   source_images=[dict(path=r["image"],sha256=r["sha256"]) for r in rows+training+summaries])
    (OUTPUT / "data-review-manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k:receipt[k] for k in ("trajectory_figures","training_figures","displayed_people","evaluated_people")},indent=2))


if __name__ == "__main__":
    build()
