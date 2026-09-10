# Landmark-selection schematic

[Editable SVG](pose_landmarks_and_laterality.svg) · [Vector PDF](pose_landmarks_and_laterality.pdf) · [Preview](pose_landmarks_and_laterality_pdf_preview.png)

The three panels draw the same 33 landmark positions. Panel A shows the full encoder-input schema: coordinate values and their validity mask, rather than the archived continuous visibility channel. Panel B highlights the twelve gait landmarks used by historical target eligibility and VICReg pooling, and panel C highlights the ten landmarks contributing to the five-pair signed target. The hips establish the pelvis reference and are not a sixth target pair. Other landmarks remain in the encoder input even when they are not highlighted; the latest motion and region policies can select different hidden tokens.

The front-facing schematic places anatomical left on the viewer's right. Coordinates are hand-drawn and do not represent a participant, detector output, or clinical gait example. The source of the point identities is the [official MediaPipe Pose landmark schema](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker); the training and target selections are those audited in the repository.

Separating these anatomical uses makes the diagram useful beside the training pipeline: it prevents a highlighted subset from being mistaken for a reduction of the dataset's 33-joint structure. The minimum label size is 20 SVG pixels at a width of 990 pixels, equivalent to 8 points when the figure occupies a 5.5-inch manuscript column.

Reproduce the vector drawing with `python make_landmark_schematic.py`. Generate its PDF with `python make_physworld_vector_pdfs.py --assets pose_landmarks_and_laterality --report-name pose_landmarks_pdf_validation.json`, using an environment with `pypdf`, Chrome or Edge, and Poppler. [Provenance](pose_landmarks_provenance.json) records all point indices; the [PDF validation](pose_landmarks_pdf_validation.json) checks dimensions, searchable labels, and vector content.
