# Photographic context for the gait-fidelity proposal

These are crops of photographs already published in open-access research figures. They provide context for the measurement problem; neither is data collected by this gait-fidelity study. The originals are retained alongside the crops. Exact crop coordinates, source URLs, file hashes, alt text, and proposed captions are recorded in `provenance.json`.

## OpenCap

`opencap-capture.jpg` (362 × 349 px) is cropped from Fig. 2 of **Uhlrich et al. (2023), “OpenCap: Human movement dynamics from smartphone videos.”** It shows walking videos inside two smartphone displays. The published figure’s existing face blur is unchanged. The source includes Scott Delp as a coauthor. This image should not imply a collaboration or endorsement.

- Article and figure source: <https://doi.org/10.1371/journal.pcbi.1011462>
- Copyright © 2023 Uhlrich et al.; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Suggested short credit: **OpenCap smartphone views. Uhlrich et al. (2023), Fig. 2; cropped, CC BY 4.0.**
- Layout: preserve the near-square aspect ratio with `object-fit: contain`; a wide crop would remove the walking participants’ feet. Avoid enlarging beyond roughly 360 CSS pixels.

## Community mobility recording

`ambient-walking.jpg` (687 × 389 px) contains two color frames from Fig. 1 of **Dawe et al. (2019), “Expanding instrumented gait testing in the community setting: A portable, depth-sensing camera captures joint motion in older adults.”** The Rush Memory and Aging Project recorded structured walking tests at participants’ residences. The participant’s existing face pixelation is unchanged, and the article explicitly documents consent to publication of these frames.

- Article and figure source: <https://doi.org/10.1371/journal.pone.0215995>
- Copyright © 2019 Dawe et al.; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Suggested short credit: **Residential mobility recording, Rush study. Dawe et al. (2019), Fig. 1; cropped, CC BY 4.0.**
- Layout: approximately 16:9; it fits a 2-inch-wide context panel without cutting off the participant’s feet.
- Describe this as **structured mobility testing in a residence**, not passive monitoring or a Stanford deployment.

## Connection to Stanford Ambient Intelligence

The [official Stanford AmI website](https://ami.stanford.edu/) describes unobtrusive multimodal sensing, everyday mobility, aging in place, a planned living-lab setting, and home-based field studies. These statements establish the research connection. Its photographs had no displayed reusable license, so they were not copied. The Rush photograph is an explicitly attributed example of the broader movement-measurement setting.

## Reproduction

The images were downloaded from the original PLOS figure endpoints on 2026-09-24. Crops use Pillow `Image.crop((left, top, right, bottom))` with the coordinates in `provenance.json`, RGB conversion, and JPEG output at quality 95 with chroma subsampling disabled. No colors, physical content, poses, or participant appearances were changed.
