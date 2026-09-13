# Study organization review — 13 September 2026

> This records the earlier documentation migration. The later [code organization](code-organization.md) renames implementation paths and preserves exact old software in a replay snapshot.

## Result and scope

Organized **296 source files** around seven scientific studies, with [motion preservation](../studies/motion-preservation/README.md) as the current objective. Each study has one overview and a small number of reading chunks. Earlier portfolios live in research-planning history; the selected proposal lives with its study and the six deferred alternatives remain in the research agenda. [Layout and naming](layout.md) explains the structure; the [ownership registry](studies.json) maps documentation to code, notebooks, tests and cluster execution.

The migration preserved the working tree that existed before this task, including untracked research work. Its local snapshot contains 799 source/workflow files. An independent audit also covered ignored source-adjacent files, for 828 original files. These are different audit scopes, not counts of experimental observations. [The migration record](migration-2026-09-13.json) lists original paths and digests, destinations, compatibility entries and protected files.

No scientific result, cohort, model API, numerical protocol or saved execution output was revised. Historical negative results remain negative. Current software validation does not establish movement preservation by a real pretrained model.

## Independent adversarial review

Three independent agents worked on distinct responsibilities. The study-layout reviewer examined scientific ownership and produced study/chunk indexes. The migration-safety reviewer independently inventoried dependencies, protected files, notebooks and local links, then checked the implemented moves. The adversarial naming reviewer challenged both the proposed taxonomy and the actual result. Follow-up tasks adapted to concrete findings: generator repair, scientific corrections, and final verification.

| Criticism or failure found | Resolution |
| --- | --- |
| Combining future-feature experiments could imply one successful distillation result. | Kept prediction gate, source scaling, target accessibility and manuscript as distinct chunks with their own controls and conclusions. Explicitly state that student benefit remains unestablished. |
| Unknown left/right correspondence and known anatomical reflection ask different questions. | Kept latent laterality separate from reflection equivariance and signed-laterality probes. |
| Broad world-model and biomechanics prompts have no single experimental owner. | Placed them with historical research planning. The current broad request remains with the research agenda. |
| A manuscript request concerned the separate laterality manuscript workspace. | Moved it to laterality development history and removed a stale future-feature manuscript pointer. |
| “Core11” could be mistaken for a standard anatomical representation. | Replaced canonical summary/proposal prose with “11-landmark lower-body representation”; retained serialized keys and runtime identifiers. Shortened the notebook name to `08_amass_training.ipynb`. |
| Old README redirects bypassed the new study overviews. | Redirected legacy FI/ICLR entry points to their owning overviews. |
| The current portfolio retained earlier advice to choose between parallel pilots. | Added an explicit current-selection banner: motion preservation selected; six alternatives deferred. |
| Moving protocols or launchers would alter sealed-run inputs or fingerprints. | Preserved all frozen implementation paths and bytes, and the five original numerical protocol files. |
| Builders still emitted old notebook names or publication destinations. | Updated generators and canonical notebook sources together; corrected evidence and manuscript output directories. |
| Longer relative paths broke figure and artifact links. | Rebased links and retained compatibility for immutable historical records. |
| Global filename uniqueness penalized conventional package-local names. | Allowed `__init__.py` and `build_notebooks.py` in separate namespaces while retaining the existing check for specialized duplicate names. |

Final independent scientific/naming review approved the organization after verifying the substantive corrections. Its two final wording corrections were also applied: a duplicated “representation” and the stale manuscript prompt statement.

## Validation

- Independent migration audit: every original file accounted for at its retained or mapped destination; moved evidence assets preserved byte-for-byte; notebook execution outputs, counts, attachments and metadata preserved.
- Frozen implementation: **79 of 79 files byte-identical** to the pre-migration working tree, including source fingerprints and launch inputs.
- Protected documents: **10 of 10 byte-identical**, including five numerical protocols, the result registry, output-organization contracts and the dated archive assessment.
- Generated notebooks: **26 of 26** agree with their authoritative builder sources. Generators were intercepted for read-only comparison; scientific notebooks were not rerun to manufacture new outputs.
- Reflection notebook startup: **six notebooks across four working-directory contexts**, all passing.
- Existing targeted suites: filename conventions **1 passed**; directory ownership **3 passed**; archived compatibility **4 passed**; motion preservation **32 passed, 1 skipped**; future-innovation tutorials **9 passed**; scaling notebooks **12 passed**. The three motion-preservation notebook tests were also run in the earlier 24-test generator check and are already included in the 32 passes.
- The skipped HumanML conversion roundtrip requires `MOTION_PRESERVATION_MOMASK_REPO`; no claim is made about that unavailable external implementation.

The registry integration test checks that declared source entry points resolve and each canonical notebook has exactly one study or shared-data owner. Both registry tests passed. Across the listed suites, 63 distinct tests passed and one was skipped. The final audit found zero migration errors and zero newly broken local links; 52 preexisting missing links remain (55 before the migration). Results and commands are recorded in the machine-readable migration record.

Run the relevant checks from the project root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_study_layout.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_unique_project_filenames.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_research_directory_ownership.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_archived_compatibility_paths.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_motion_preservation_*.py' -v
PYTHONPATH=src .venv/bin/python -m gavd6_sjepa.workspace_validation.notebook_startup_validation
git diff --check
```

## Evidence and portability limits

The independent audit distinguishes broken links present before this migration from newly introduced ones. Historical links to unavailable run bundles remain evidence-location gaps; placeholder results were not created to make a link check pass. The earlier assessment recorded 16 absent registered canonical/legacy run locations and the missing GAVD source-reservation CSV. No remote storage audit or new model experiment was performed here.

The pre-migration snapshot and detailed local verifier reports remain under `work/repository-organization/20260913/`. They include preexisting untracked material and are local recovery records. Committed navigation uses relative paths; older notebook and asset paths use relative symlinks, so checkouts must preserve symlinks for those compatibility paths.
