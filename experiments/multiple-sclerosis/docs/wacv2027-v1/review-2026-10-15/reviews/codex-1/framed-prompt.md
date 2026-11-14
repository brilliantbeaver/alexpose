<!--
Prompt frame for a raw `codex exec` review leg. The lead substitutes
2c7739f73e392d3e6765bba08b2d430f8df9f82d, 828b78bafe77934ffb75ab7ec3d4e12c7e8177a1 and First-party scientific manuscript review, not a software merge. The change is three documents under revision/. The original supplied paper text is docs/wacv2007/review-2026-10-15/working/original-paper.txt; source evidence is in notebooks, sjepa code, saved artifacts and docs/wacv2007/review-2026-10-15/evidence/. Review all three proposed documents against primary evidence and the original. Newly added documents do not have base versions; use the original paper and base evidence for comparison. Do not change any files or run scripts/tests/builds, do not delegate recursively, and do not query MCP, web or memory. Use read-only shell operations (git, nl, cat, rg). Read files with nl -ba, not bare sed.

Challenge these numbered properties: (1) Every replacement numerical claim matches a coherent corrected October execution; historical RF .411 is fairly disclosed and no causal S-JEPA improvement is fabricated. (2) The positive representation-learning claim is justified while preserving lack of untrained/temporal/masking controls and strong mean-pose result. (3) Methods match code/notebooks: sample sizes, use of labels, restart, masking denominator, readout, input channels, evaluation unit. (4) Copy-paste instructions have precise replacement boundaries, preserve essential original citations/methods, and figures/tables/text agree. (5) Main prose is understandable to educated non-ML researchers, avoids repeated statistical defeat framing without hiding outcomes, and remains publication-appropriate. (6) Evidence/venue/provenance claims are no stronger than saved support. Name any property claimed above that no check actually exercises, and any check that passes for the wrong reason. Assess limitations of metric reproduction vs retraining, and lack of clinical validation, without demanding new experiments as a condition of a bounded observational result.

Known fixes: RF corrected at ambient/classification/features.py left_ankle_range; updated all RF scores, folds, confusion entries, bootstrap. Mask percentage denominator fixed. Fold0 notebook03 checkpoint payload matches current fold0 state per host verification (diagnostic means remain saved notebook outputs, not recovered history). Source IDs are not participants; all33 landmarks vs lower-limb RF noted. R15 instructions now preserve intervening extraction/normalization. Do not invent findings. For each claim say HOLDS/BROKEN/NOT REACHED with quoted supporting passage or code and exact file:line. Include evidence anchors for each of the three revision files: line count and verbatim final line. Findings need severity, trigger, consequence, location, and concrete fix. Distinguish inherited issues from introduced errors. End with exactly Verdict: approve or Verdict: needs-attention. No praise or generic summary. (see the launch section in SKILL.md)
and feeds the result on stdin. Structure modelled on the adversarial-review
prompt of the OpenAI codex Claude Code plugin (Apache-2.0); the text here
is this suite's own.
-->
You are reviewing a change that someone wants to merge. Your job is to find
the strongest reasons it should NOT merge yet. You are not here to confirm
that it works, to summarize it, or to credit good intentions.

## Target

- Range: `2c7739f73e392d3e6765bba08b2d430f8df9f82d..828b78bafe77934ffb75ab7ec3d4e12c7e8177a1` in the current repository. List it with
  `git diff --stat 2c7739f73e392d3e6765bba08b2d430f8df9f82d 828b78bafe77934ffb75ab7ec3d4e12c7e8177a1` and read it with
  `git diff 2c7739f73e392d3e6765bba08b2d430f8df9f82d 828b78bafe77934ffb75ab7ec3d4e12c7e8177a1`.
- Read-only means reading: use git, grep, nl and cat only. Do not run
  tests, builds or scripts, do not call MCP or other external tools, and do
  not write anywhere, a temporary directory included.

## Lens for this round

First-party scientific manuscript review, not a software merge. The change is three documents under revision/. The original supplied paper text is docs/wacv2007/review-2026-10-15/working/original-paper.txt; source evidence is in notebooks, sjepa code, saved artifacts and docs/wacv2007/review-2026-10-15/evidence/. Review all three proposed documents against primary evidence and the original. Newly added documents do not have base versions; use the original paper and base evidence for comparison. Do not change any files or run scripts/tests/builds, do not delegate recursively, and do not query MCP, web or memory. Use read-only shell operations (git, nl, cat, rg). Read files with nl -ba, not bare sed.

Challenge these numbered properties: (1) Every replacement numerical claim matches a coherent corrected October execution; historical RF .411 is fairly disclosed and no causal S-JEPA improvement is fabricated. (2) The positive representation-learning claim is justified while preserving lack of untrained/temporal/masking controls and strong mean-pose result. (3) Methods match code/notebooks: sample sizes, use of labels, restart, masking denominator, readout, input channels, evaluation unit. (4) Copy-paste instructions have precise replacement boundaries, preserve essential original citations/methods, and figures/tables/text agree. (5) Main prose is understandable to educated non-ML researchers, avoids repeated statistical defeat framing without hiding outcomes, and remains publication-appropriate. (6) Evidence/venue/provenance claims are no stronger than saved support. Name any property claimed above that no check actually exercises, and any check that passes for the wrong reason. Assess limitations of metric reproduction vs retraining, and lack of clinical validation, without demanding new experiments as a condition of a bounded observational result.

Known fixes: RF corrected at ambient/classification/features.py left_ankle_range; updated all RF scores, folds, confusion entries, bootstrap. Mask percentage denominator fixed. Fold0 notebook03 checkpoint payload matches current fold0 state per host verification (diagnostic means remain saved notebook outputs, not recovered history). Source IDs are not participants; all33 landmarks vs lower-limb RF noted. R15 instructions now preserve intervening extraction/normalization. Do not invent findings. For each claim say HOLDS/BROKEN/NOT REACHED with quoted supporting passage or code and exact file:line. Include evidence anchors for each of the three revision files: line count and verbatim final line. Findings need severity, trigger, consequence, location, and concrete fix. Distinguish inherited issues from introduced errors. End with exactly Verdict: approve or Verdict: needs-attention. No praise or generic summary.

## How to review

1. Read with line numbers so every citation is exact: `nl -ba <path>` for
   the new version and `git show 2c7739f73e392d3e6765bba08b2d430f8df9f82d:<path> | nl -ba` for the old one.
2. For every file the range touches, read the version at `2c7739f73e392d3e6765bba08b2d430f8df9f82d` as well
   as the new one. Many defects are only visible as a difference: something
   the old code did that the new code silently stopped doing, or a state the
   old code never wrote.
3. For each defect, check whether it already exists at `2c7739f73e392d3e6765bba08b2d430f8df9f82d`
   (`git show`, `git blame`) and say which. Report inherited defects too,
   labelled inherited: they matter when they break something the lens or
   the change claims, or when the change now relies on them. Only an
   inherited defect that touches neither is a `nit`.
4. Trace each hazard to the callers or consumers of the changed code and
   say whether a later check already stops it. If one does, lower the
   severity and say so; do not drop the finding.
5. For each test the change adds or edits, ask whether it could fail if the
   code it guards were broken. A test that passes either way is a finding.
6. Compare what the docs, README, comments and commit message promise with
   what the code does. A promise the code does not keep is a finding.
7. A diff cannot show a binary file's content. Compare blob ids
   (`git ls-tree 2c7739f73e392d3e6765bba08b2d430f8df9f82d <path>` against `828b78bafe77934ffb75ab7ec3d4e12c7e8177a1`) before saying a binary
   did or did not change.
8. Default to doubt. Something that only works on the happy path is a
   weakness. "Will be fixed later" does not count.

## Where expensive failures hide

- Trust boundaries: input from users, other processes, files, the network
  or the environment reaching code that assumes it is safe; permissions;
  injection into shells, SQL or option parsers.
- Filesystem: symlinks, TOCTOU races, partial writes, permissions, files
  opened for writing that were never written before.
- State: data loss, duplication, corruption, irreversible changes, anything
  a crash halfway through leaves behind.
- Failure paths: retries, timeouts, partial failure, idempotency, cleanup
  that runs or does not run, errors that are swallowed.
- Concurrency and ordering: races, stale reads, re-entrancy, assumptions
  about which event arrives first.
- External tools and versions: flags or behaviour that differ between
  versions, and configuration that changes what a command does.
- Defaults and empty cases: an empty list, a missing key, zero, an unset
  variable, the first run on a new machine.

## What counts as a finding

- It names a location as `path:line` in `828b78bafe77934ffb75ab7ec3d4e12c7e8177a1`, or in `2c7739f73e392d3e6765bba08b2d430f8df9f82d` when
  the defect is something the change removed. A missing piece of handling
  is cited at the `path:line` where that handling belongs.
- It describes a concrete sequence that leads to a wrong result, including
  the input or state that triggers it.
- It is about behaviour, not style. Leave out preferences and naming.
- Say how sure you are. If you find nothing that meets this bar, say so
  plainly. Do not invent findings to fill the report.

Severity:
- `blocker`: the change must not merge while this stands (wrong results,
  data loss, a security or safety hole, a broken main path).
- `should-fix`: a real defect with limited reach or a workaround; fix it in
  this change unless there is a stated reason not to.
- `nit`: minor, worth mentioning, never a reason to hold the merge.

## Output

1. If the lens poses numbered claims or questions, answer each one first:
   `HOLDS`, `BROKEN` or `NOT REACHED`, with the evidence. Something you
   could not confirm is `NOT REACHED`, not a finding.
2. Then the numbered findings, most severe first. For each: severity,
   `path:line`, whether it is introduced or inherited, the failing
   sequence, and the fix.
3. The verdict is `approve` when there is no `blocker` and no `should-fix`
   finding, and `needs-attention` otherwise. The last line of your answer is
   exactly `Verdict: approve` or `Verdict: needs-attention`, written as
   plain text: no bold, no heading, no backticks.
