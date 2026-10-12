# Workflow and scheduler implementation review

Review date: 22 September 2026. Scope: HAIC orchestration, source-release transfer, configuration reuse, notebook tutorials and local execution. This review does not claim that the new source study has run on HAIC.

The workflow author independently reviewed the root agent's scheduler, configuration, profiling and CLI implementation. The root agent reviewed the workflow integration, and the data agent was asked to review the rendered notebook mask figure. The workflow tests were written by the workflow author and are identified as implementation checks rather than independent scientific replication.

## Findings and corrections

| Finding | Why it mattered | Final correction |
| --- | --- | --- |
| Batch scripts initially located helpers relative to their own filename. | Slurm executes a copy in its spool directory, which does not contain the release helpers. | Workers and coordinator locate the helper through the saved `GF_ROOT`; a test executes relocated batch-script copies. |
| A successful response or a transport error from `sbatch` could leave an uncertain job identity. | A second launch could create a duplicate coordinator or worker and spend the same apparent budget twice. | Unique names and reservations are persisted before submission. Active jobs are reused; uncertain responses block duplicate submission pending scheduler reconciliation. |
| Worker recovery removed the UTC offset from the submission timestamp. | A cluster using another local timezone could search the wrong accounting interval. | Unique-name recovery invokes accounting with `TZ=UTC`. |
| Worker submission relied on scheduler defaults. | Site defaults or requeue behavior could change allocation shape and attempt ownership. | Submission explicitly sets one node, one task, environment export and no requeue. |
| Merged preparation needed its own immutable identity. | A later worker could read a changed merged bundle while individual shard receipts remained valid. | The merged manifest is hashed and checked; loading verifies the input and target array hashes. |
| The initializer could fall back to the earlier 2+2 pilot roster. | A technically complete run could be mistaken for the planned full population. | Source initialization requires the retained 24-training/8-development roster. |
| Reinitializing a run could silently ignore different explicit asset arguments. | The user could believe a new source bundle or checkout had been selected. | Conflicting explicit asset-root, preparation-config and source-bundle overrides are rejected. |
| CPU evaluation shares the coordinator allocation. | An 8 GB controller allowance was small for the expanded tables. | The coordinator requests 32 GB; actual source memory use still requires HAIC measurement. |
| Headless plotting did not display the mask comparison through `plt.show()`. | Notebook execution could pass while its explanatory figure was absent. | The tutorial explicitly displays generated PNG bytes and closes the figure. |
| The evaluation scatter overlaid method labels when their scores were close. | Coincident fixture scores made the figure difficult to read and clipped annotations. | Two horizontal point panels now show one labelled row per method, with figure height scaled to the number of methods and explicit lower-is-better axes. |

The runtime profile uses timing rather than model ranking to choose the full or half-budget schedule across the entire matrix. It does not reuse profile checkpoints for final fits. Its short-run extrapolation is labelled as an estimate, while global allocation reservations still enforce the actual study ceiling.

## Workflow checks

Nine workflow tests pass, covering shell syntax, notebook code compilation, all twelve tutorial files, relocated Slurm scripts, active-controller idempotence, ambiguous submission responses, transport failures, finished-job accounting, rejection of fixture GPU submission and unsafe remote paths. The three new navigation/run documents have no missing local links. A local release-packaging preview completed without contacting HAIC or transferring experiment data.

The notebook test runner executes real Jupyter kernels against a fresh CPU fixture and saves the executed copies. The local sandbox blocks Jupyter's loopback sockets, so a bounded local-process escalation was used for this validation. No external messages, data transfer, HAIC connection or GPU allocation were part of that check.

An earlier tutorial pass was deliberately interrupted when the independent scientific review strengthened per-update endpoint matching. Its child process had completed 135 optimization phases and exited; it is excluded from final validation. A later complete pass exposed the crowded evaluation figure during visual review. The final pass uses another fresh work directory and includes the corrected figure; its numerical tables and report text are byte-identical to the preceding completed run.

## Final notebook execution

Final run location: `/private/tmp/gf-notebook-final-v2-20260922/work`. Executed notebook copies and the machine-readable execution receipt are under `/private/tmp/gf-notebook-final-v2-20260922/executed`.

The final pass completed at **2026-09-22 07:18:44 UTC**. All **12 notebooks and 53 code cells passed** with zero error outputs. Shared preparation and all **135 optimization phases producing 102 models** completed without a failed attempt. Notebook 06 returned `GAIT_FIDELITY_VERIFIED`, checking **915 retained artifacts** and reconstructing **39,168 neural per-window metric rows**. Two PNG outputs were embedded in the executed notebooks.

The evaluation identifies the output as `SOFTWARE_FIXTURE_COMPLETE`, with `fixture-tested` evidence, no independent confirmation and no clinical validation. The machine-readable receipt is `/private/tmp/gf-notebook-final-v2-20260922/executed/execution.json`. It records the duration and passing status of every notebook; these local CPU durations are not an estimate of source-study GPU throughput.

The mask comparison generated by notebook 02 was visually inspected at its rendered resolution and independently accepted by the data agent. All five policies, twelve joint labels and the hidden/visible key are legible, with separate axes and no overlap or clipping. The figure uses the actual training sampler and illustrates one stochastic draw; anatomical connectivity and marginal coverage are established by the graph and mask audits rather than by this image alone. The prepared interactive viewer is retained at `work/data/viewer.html` for the same fixture; its examples do not substitute for source video review.

The revised evaluation figure was independently accepted by both the root reviewer and the data agent, and its final regenerated PNG was inspected again after the last execution. All 40 method labels, both panel titles and axes remain readable without collisions or clipping. It reports descriptive means and directs readers to the separate uncertainty results. The figure must retain the fixture context supplied by its report when shared; its values do not constitute source-study findings.

## Limits of this validation

The CPU fixture checks orchestration, phase dependencies, losses, output contracts and notebook execution at a small update budget. It does not certify H100 rendering, MMPose runtime behavior on HAIC, dataset quality, cluster wall-time limits, source throughput or scientific effectiveness. The HAIC workflow checks its actual CUDA stack inside a preparation worker and retains that result before rendering. Full-interval reference review and any independent confirmation population remain separate scientific requirements.
