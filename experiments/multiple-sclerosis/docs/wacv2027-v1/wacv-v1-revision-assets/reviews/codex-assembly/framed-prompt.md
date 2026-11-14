<!--
Prompt frame for a raw `codex exec` review leg. The lead substitutes
0c1985cfaa70fa14c1486f80bd8641fadee63f2e, 50a1b14b500b6876aa7f10fc046c071024547a80 and First-party scientific manuscript assembly review. User approved implementing the previously reviewed revisions to the original PDF. New target files are the complete main TeX, supplement TeX, bibliography and figure/assembly sources under docs/wacv2027-v1. Base contains original extracted paper and prior independently checked evidence and replacement fragments. Added files have no base versions; compare them to the original and retained fragments. No recursive delegation or CLI, no MCP/web/memory calls, no scripts/tests/builds, no edits. Use read-only git, nl, rg, cat. Read with nl -ba for exact citations.

Challenge: (1) Are the assembled manuscript and supplement scientifically coherent and faithful to verified results and implementation? (2) Are the trained-representation claims appropriately bounded, with mean pose, validation size, corrected and historical RF, and no causal pretraining/clinical novelty exaggeration? (3) Do assembly additions silently lose methods or citations, contradict the prior corrections, or introduce unverifiable claims? (4) Do plotted values, scales, denominators and labels agree with manuscript and evidence? (5) Do the source files have obvious build/reproducibility defects within scope? Figure generation can fall back to supplied figure-values.json; build_manuscript.py is an author-workspace helper, not an Overleaf requirement. Delivered expanded TeX requires only the matching .bib, official wacv.sty/.bst and the three figure PDFs. These dependencies and final rendering are checked separately by the host and are omitted from this text-review snapshot; do not report their omission here as a broken delivery. No claim of a standalone raw-video/model reproduction packet is made.

Known fixes before this frozen target: exact 43.75% mean-pose Normal recall formatted 43.8 despite serialized float noise; mask overshoot attributed to whole final region not overlap itself; reflection described as untested by a controlled comparison, not proven error; complete untransformed target input and initialization restored; R14 draw layout fully specified. Main is 8 PDF pages with body ending 7; supplement 3 pages. Host inspected rendered pages, no overflow/unresolved refs. No author ID supplied; DRAFT is intentionally identified in TeXcomment,not assigned ID. Ethics/permissions unknown and not invented; authors must supply supported facts before submission. Do not demand new experiments as a condition for the bounded observed result, but flag real unsupported assertions.

For each numbered claim answer HOLDS/BROKEN/NOT REACHED with quoted source and exact line. Name any property claimed above that no check actually exercises, and any check that passes for the wrong reason. Findings need severity, introduced/inherited, trigger, consequence, location and concrete fix. Supply line counts and verbatim last lines for both main and supplement as read anchors. No praise or generic summary; do not invent findings. End exactly Verdict: approve or Verdict: needs-attention. (see the launch section in SKILL.md)
and feeds the result on stdin. Structure modelled on the adversarial-review
prompt of the OpenAI codex Claude Code plugin (Apache-2.0); the text here
is this suite's own.
-->
You are reviewing a change that someone wants to merge. Your job is to find
the strongest reasons it should NOT merge yet. You are not here to confirm
that it works, to summarize it, or to credit good intentions.

## Target

- Range: `0c1985cfaa70fa14c1486f80bd8641fadee63f2e..50a1b14b500b6876aa7f10fc046c071024547a80` in the current repository. List it with
  `git diff --stat 0c1985cfaa70fa14c1486f80bd8641fadee63f2e 50a1b14b500b6876aa7f10fc046c071024547a80` and read it with
  `git diff 0c1985cfaa70fa14c1486f80bd8641fadee63f2e 50a1b14b500b6876aa7f10fc046c071024547a80`.
- Read-only means reading: use git, grep, nl and cat only. Do not run
  tests, builds or scripts, do not call MCP or other external tools, and do
  not write anywhere, a temporary directory included.

## Lens for this round

First-party scientific manuscript assembly review. User approved implementing the previously reviewed revisions to the original PDF. New target files are the complete main TeX, supplement TeX, bibliography and figure/assembly sources under docs/wacv2027-v1. Base contains original extracted paper and prior independently checked evidence and replacement fragments. Added files have no base versions; compare them to the original and retained fragments. No recursive delegation or CLI, no MCP/web/memory calls, no scripts/tests/builds, no edits. Use read-only git, nl, rg, cat. Read with nl -ba for exact citations.

Challenge: (1) Are the assembled manuscript and supplement scientifically coherent and faithful to verified results and implementation? (2) Are the trained-representation claims appropriately bounded, with mean pose, validation size, corrected and historical RF, and no causal pretraining/clinical novelty exaggeration? (3) Do assembly additions silently lose methods or citations, contradict the prior corrections, or introduce unverifiable claims? (4) Do plotted values, scales, denominators and labels agree with manuscript and evidence? (5) Do the source files have obvious build/reproducibility defects within scope? Figure generation can fall back to supplied figure-values.json; build_manuscript.py is an author-workspace helper, not an Overleaf requirement. Delivered expanded TeX requires only the matching .bib, official wacv.sty/.bst and the three figure PDFs. These dependencies and final rendering are checked separately by the host and are omitted from this text-review snapshot; do not report their omission here as a broken delivery. No claim of a standalone raw-video/model reproduction packet is made.

Known fixes before this frozen target: exact 43.75% mean-pose Normal recall formatted 43.8 despite serialized float noise; mask overshoot attributed to whole final region not overlap itself; reflection described as untested by a controlled comparison, not proven error; complete untransformed target input and initialization restored; R14 draw layout fully specified. Main is 8 PDF pages with body ending 7; supplement 3 pages. Host inspected rendered pages, no overflow/unresolved refs. No author ID supplied; DRAFT is intentionally identified in TeXcomment,not assigned ID. Ethics/permissions unknown and not invented; authors must supply supported facts before submission. Do not demand new experiments as a condition for the bounded observed result, but flag real unsupported assertions.

For each numbered claim answer HOLDS/BROKEN/NOT REACHED with quoted source and exact line. Name any property claimed above that no check actually exercises, and any check that passes for the wrong reason. Findings need severity, introduced/inherited, trigger, consequence, location and concrete fix. Supply line counts and verbatim last lines for both main and supplement as read anchors. No praise or generic summary; do not invent findings. End exactly Verdict: approve or Verdict: needs-attention.

## How to review

1. Read with line numbers so every citation is exact: `nl -ba <path>` for
   the new version and `git show 0c1985cfaa70fa14c1486f80bd8641fadee63f2e:<path> | nl -ba` for the old one.
2. For every file the range touches, read the version at `0c1985cfaa70fa14c1486f80bd8641fadee63f2e` as well
   as the new one. Many defects are only visible as a difference: something
   the old code did that the new code silently stopped doing, or a state the
   old code never wrote.
3. For each defect, check whether it already exists at `0c1985cfaa70fa14c1486f80bd8641fadee63f2e`
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
   (`git ls-tree 0c1985cfaa70fa14c1486f80bd8641fadee63f2e <path>` against `50a1b14b500b6876aa7f10fc046c071024547a80`) before saying a binary
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

- It names a location as `path:line` in `50a1b14b500b6876aa7f10fc046c071024547a80`, or in `0c1985cfaa70fa14c1486f80bd8641fadee63f2e` when
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
