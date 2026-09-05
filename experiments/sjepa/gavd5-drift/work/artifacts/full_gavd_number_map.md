# Full-GAVD re-scope number map (source of truth)

Generated 2026-09-02. Two-stage validity check:
1. `yt-dlp --simulate` existence sweep over all 103 videos.
2. Full download + frame-span validation via notebook 01 (must reach the annotated last frame).

Artifacts: `gavd_full_validity_report.json`, `gavd_valid_sequences.csv`.

## Canonical corpus = DOWNLOAD-VALID set (use these everywhere)

| item | OLD curated | full valid |
|---|---:|---:|
| total gait sequences | 96 | **642** |
| total unique source videos | 18 | **94** |
| normal sequences / videos | 12 / 1 | **284 / 30** |
| parkinsons sequences / videos | 9 / 2 | **42 / 9** |
| stroke sequences / videos | 12 / 3 | **75 / 18** |
| myopathic sequences / videos | 47 / 10 | **183 / 28** |
| cerebralpalsy sequences / videos | 16 / 2 | **58 / 9** |

Raw `data-gavd`: 666 seq / 103 videos. Existence-valid (yt-dlp simulate): 645 / 96.
Download-valid (fetchable + reaches annotated frame): **642 / 94** — the canonical number.

## 9 excluded videos

Existence-invalid (7, yt-dlp): sf5X4YYkWUA, WWS-iOlLsoo (cerebralpalsy); yULxvDc9e8c (myopathic);
OoCDFmCm1DE, yFBy0X0D-w8 (normal); dxRMtNtjwCc (parkinsons); YjRoLtP1di0 (stroke).
Download-invalid (2): JUMhhwFANKE (parkinsons — video unavailable at download);
n93bgWhLZk4 (myopathic — no obtainable format reaches the annotated frame span).

## DERIVED numbers — DO NOT substitute (regenerated in Phase B re-execution)

159/35 curriculum, 75 stage-0 normal, 64/63/17-added augmentation, all classifier/probe
metrics, train/test split sizes, figure values. Leave; mark "pending regeneration".

## Overloaded — DO NOT touch

12-landmark whitelist, 33 landmarks, 64 frames, 16 time patches, 528 tokens, 4 frames/patch,
embedding width 96, 8×12=96 tokens, thresholds, epochs, seeds. (Note: 96 as embedding-width or
token-count is NOT the video count — do not change those.)
