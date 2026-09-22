# HAIC dataset-availability review

21 September 2026. An independent data-plan reviewer inspected the main proposal and `data/README.md`, `data/availability.md`, `data/clinical-candidates.md` and `data/local-videos.md`. The reviewer did not author these availability revisions or edit them during review. This record summarizes the returned findings and the coordinating author's corrections.

**Verdict: passes availability review, with no blocking issues.** The exact AMASS, GAVD, COCO, body-model, DMPL, texture, UV-map and background paths match the user's supplied HAIC environment. Completed source-run records corroborate the synthetic assets actually used, while the prose does not claim a fresh remote filesystem audit or complete coverage of every release.

The classification separates assets already provisioned on HAIC, the 91 clips verified on the Mac, and optional datasets not yet acquired for this study. The HAIC copy of the local clip collection remains unconfirmed. Optional acquisition does not block synthetic development, and existing files are distinguished from the new annotations, intervention pairs and confirmation roster still required.

The reviewer also checked that GAVD labels were not treated as independent movement references, that shared source IDs did not establish exact clip availability or external independence, and that AMASS BMLmovi motion did not imply possession of the full MoVi video/reference release. Local links in the reviewed documents resolved.

Two minor wording suggestions were incorporated: optional sources must support the intended **experiment**, because stroke motion may also enter fitting; and the healthy MoVi release is described as a **candidate dataset**, rather than as a clinical cohort.

The generated HTML was rebuilt from the revised README. Existing SVGs, previews and interactive JavaScript/CSS remain unchanged. The [availability validation receipt](../records/availability-validation-20260921.json) records link, syntax and preservation checks. The earlier paper reviews and validation remain dated snapshots of their reviewed versions. This update did not connect to HAIC, download a dataset, transfer local videos or run a new experiment.
