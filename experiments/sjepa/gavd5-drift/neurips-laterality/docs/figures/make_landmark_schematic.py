"""Draw the 33-point schema and the two distinct anatomical selections.

Positions are hand-drawn, not extracted observations. No model or participant
data are loaded. Point identities follow MediaPipe's published Pose schema.
"""
from __future__ import annotations

import json
from pathlib import Path

from make_physworld_figures import SVG

HERE = Path(__file__).resolve().parent
BLUE = "#176FA1"
AMBER = "#B96A20"
INK = "#183647"
MUTED = "#526873"
GRAY = "#CED8DE"
LINK = "#B9C8D0"

# Front-facing schematic: the person's anatomical left is on the viewer's right.
POINTS = {
    0: (0, .09), 1: (.055, .055), 2: (.083, .050), 3: (.112, .055),
    4: (-.055, .055), 5: (-.083, .050), 6: (-.112, .055),
    7: (.155, .092), 8: (-.155, .092), 9: (.052, .140), 10: (-.052, .140),
    11: (.215, .255), 12: (-.215, .255),
    13: (.290, .435), 14: (-.290, .435),
    15: (.305, .610), 16: (-.305, .610),
    17: (.345, .660), 18: (-.345, .660),
    19: (.310, .682), 20: (-.310, .682),
    21: (.274, .625), 22: (-.274, .625),
    23: (.140, .535), 24: (-.140, .535),
    25: (.146, .760), 26: (-.146, .760),
    27: (.142, .960), 28: (-.142, .960),
    29: (.101, 1.020), 30: (-.101, 1.020),
    31: (.237, 1.045), 32: (-.237, 1.045),
}
LEFT = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT = set(POINTS) - LEFT - {0}
GAIT = {11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
TARGET = GAIT - {23, 24}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
         (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
         (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
         (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
         (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]


def body(svg, center, top, selected, pelvis=False):
    coords = {index: (center + x * 310, top + y * 340) for index, (x, y) in POINTS.items()}
    for a, b in EDGES:
        svg.line(*coords[a], *coords[b], LINK, 3)
    for index, (x, y) in coords.items():
        color = BLUE if index in LEFT else AMBER if index in RIGHT else MUTED
        active = index in selected
        svg.circle(x, y, 5.7 if active else 4.1, color if active else GRAY,
                   "white", 1.4)
    if pelvis:
        for index in (23, 24):
            svg.circle(*coords[index], 8, "white", MUTED, 2)
        mid_y = coords[23][1]
        svg.circle(center, mid_y, 3.4, MUTED)
        svg.text(center, mid_y - 16, "pelvis", 20, MUTED, anchor="middle")
    return coords


def main():
    assert len(POINTS) == 33 and len(GAIT) == 12 and len(TARGET) == 10
    svg = SVG(990, 940, "One landmark array, two anatomical selections",
              "Three front-facing schematic skeletons show all 33 MediaPipe positions, "
              "12 gait landmarks used by historical target eligibility and VICReg pooling, "
              "and 10 landmarks in the five-pair raw observed laterality target. "
              "No landmarks are deleted from the input; hips define the pelvis reference.")
    svg.text(28, 44, "One landmark array, two anatomical selections", 30, INK, 700)
    svg.text(28, 80, "Highlighting selects a use for a point; all 33 positions stay in the input.", 21, MUTED)
    definitions = [
        (24, "A  |  Full input", "33 landmarks", set(POINTS), False,
         ["Coordinates + validity", "retain named joint identities."]),
        (350, "B  |  Training anatomy", "12 gait landmarks", GAIT, False,
         ["Shoulders, hips, knees,", "ankles, heels, foot tips."]),
        (676, "C  |  Signed target", "10 target landmarks", TARGET, True,
         ["Five left/right pairs;", "hips supply a reference."]),
    ]
    for x, title, subtitle, selected, pelvis, detail in definitions:
        svg.rect(x, 110, 290, 536, "#F6F9FB", "#D7E2E8", 16)
        svg.text(x + 18, 148, title, 22, INK, 700)
        svg.text(x + 18, 180, subtitle, 21, MUTED)
        body(svg, x + 145, 205, selected, pelvis)
        svg.text(x + 18, 599, detail, 20, MUTED, leading=1.4)
    svg.circle(38, 680, 6.5, BLUE)
    svg.text(53, 687, "Anatomical left", 20, INK)
    svg.circle(263, 680, 6.5, AMBER)
    svg.text(278, 687, "Anatomical right", 20, INK)
    svg.circle(498, 680, 6.5, GRAY)
    svg.text(513, 687, "Unselected, still present", 20, INK)
    svg.text(28, 724, "Front view: the person's left appears on the viewer's right.", 20, MUTED)
    svg.rect(24, 755, 942, 157, "#EEF5F7", "#CBDDE4", 14)
    svg.text(44, 788, "The two selections answer different questions", 22, INK, 700)
    svg.text(44, 821, [
        "Training: historical target eligibility and VICReg use the 12 gait points.",
        "Latest motion / region samplers can select other body tokens.",
        "Target: use observed timestamps before interpolation or 64-frame resizing.",
    ], 21, MUTED, leading=1.42)
    svg.save("pose_landmarks_and_laterality.svg")
    provenance = {
        "asset": "pose_landmarks_and_laterality.svg", "kind": "hand-drawn schematic",
        "measured_participant_data": False, "model_predictions": False,
        "schema_source": "https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker",
        "all_landmark_indices": sorted(POINTS), "training_selection_indices": sorted(GAIT),
        "target_selection_indices": sorted(TARGET), "pelvis_reference_indices": [23, 24],
        "interpretation": "Training selection depicts historical target eligibility and the twelve-landmark VICReg pool, not every latest mask policy.",
        "minimum_font_pixels": 20, "minimum_font_points_at_5_5_inches": 8.0,
    }
    (HERE / "pose_landmarks_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
