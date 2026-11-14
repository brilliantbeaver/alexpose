# Final manuscript package

This is revision 3 of **Learning Skeleton Representations from a Small Collection of Neurological Gait Videos**, prepared for WACV 2027 Vision4Neuro. The manuscript is anonymous and uses the unchanged official WACV 2027 review style. The eight-page PDF contains the main text through page 7 and references on pages 7–8; the separate methods supplement has three pages. The workshop allows eight main-paper pages excluding references for regular papers.

## Open in Overleaf

Create a new Overleaf project by uploading `wacv2027-final-overleaf.zip`. Select `main.tex` as the main document and XeLaTeX as the compiler, then recompile. All bibliography, style and figure dependencies are included. The package was rebuilt locally from an extracted ZIP with Tectonic, which uses the XeTeX engine; it was not uploaded to Overleaf. To compile the companion, select `supplementary.tex` as the main document temporarily. The official kit recommends pdfLaTeX as another build option, but that engine was not available for local verification.

Local builds with Tectonic:

```sh
tectonic main.tex
tectonic supplementary.tex
python3 reproducibility/verify_metrics.py
```

The included `main.pdf` and `supplementary.pdf` are the visually inspected final renderings. The main TeX is byte-for-byte revision 3. The two vector PDF figures have readable labels, straight connectors and no rasterized text. The reproduction packet checks stored predictions and source partitions; it does not retrain the model or release source videos.

## Submission details to complete

Replace `DRAFT` in `\def\wacvPaperID{DRAFT}` in both TeX files with the actual assigned submission identifier. Retain anonymous author information for review. The style's Applications label describes the paper format; choose **Vision4Neuro** in the workshop submission system. The actual call controls the submission route and current dates: https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html . Nothing has been submitted by preparing this package.

The manuscript states the unresolved data-use issues explicitly. Authors still need to establish the appropriate source-media permissions, secondary-use/consent basis and ethics determination. The retained records do not verify diagnoses or unique participants across recordings. Do not replace these statements with unsupported approvals. The paper reports a useful learned-feature comparison on this collection, without claiming clinical validation or an isolated causal gain from pretraining.

Private revision scores, adversarial prompts, notebook extracts and local audit paths are deliberately outside this anonymous package. They remain in the working revision directory. The accompanying metric packet contains coded recording and clip identifiers, with no source URLs, participant names or identifier mapping.
