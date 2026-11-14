<!--
Prompt frame for a raw `codex exec` review leg. The lead substitutes
99365f9f96059dcb246dbda1d65151fe38b126a5, d4b546a22011660e1ddabe7237dd9075dae04eb9 and First-party scientific manuscript challenge review, not a code merge. The user requests a WACV 2027 Vision4Neuro paper derived from the seven top-level notebooks, which are the source of truth. The original supplied PDF was converted to original-draft.txt. Review manuscript.tex and bibliography against notebook text, implementation, OOF artifacts, and evidence audits. Do not treat later RF rerun as the main notebook study. No edits, recursive Codex, MCP, or external tools. Only read files, using nl -ba for exact citations. Ask whether the inference is justified, not merely whether text repeats code. Scope is a bounded empirical pilot, not clinical deployment. For each question give HOLDS, BROKEN or NOT REACHED with quotes: (1) every numerical claim, denominator, weighting, bootstrap and comparison is faithful; (2) Section 4.1 masking, 12 landmarks/all33, coordinate chain and teacher readout are accurate; (3) same-seed runs, source vs participant grouping, train/validation boundaries and notebook vs additional reanalysis are distinguished; (4) citations support scientific claims and do not hallucinate; (5) negative-result wording neither establishes equivalence nor causal temporal benefit and identifies consequential limitations. Name claimed properties for which the retained experiments provide no evidence. Prioritize concrete scientific or numerical defects and missing reproduction details, not generic wish lists for new experiments. Acknowledge known limitations already stated, do not report them as undisclosed. Report line count and last line of manuscript.tex as reading anchor. Each defect requires exact file:line, consequence, fix and severity. End exactly Verdict: approve or Verdict: needs-attention. (see the launch section in SKILL.md)
and feeds the result on stdin. Structure modelled on the adversarial-review
prompt of the OpenAI codex Claude Code plugin (Apache-2.0); the text here
is this suite's own.
-->
You are reviewing a change that someone wants to merge. Your job is to find
the strongest reasons it should NOT merge yet. You are not here to confirm
that it works, to summarize it, or to credit good intentions.

## Target

- Range: `99365f9f96059dcb246dbda1d65151fe38b126a5..d4b546a22011660e1ddabe7237dd9075dae04eb9` in the current repository. List it with
  `git diff --stat 99365f9f96059dcb246dbda1d65151fe38b126a5 d4b546a22011660e1ddabe7237dd9075dae04eb9` and read it with
  `git diff 99365f9f96059dcb246dbda1d65151fe38b126a5 d4b546a22011660e1ddabe7237dd9075dae04eb9`.
- Read-only means reading: use git, grep, nl and cat only. Do not run
  tests, builds or scripts, do not call MCP or other external tools, and do
  not write anywhere, a temporary directory included.

## Lens for this round

First-party scientific manuscript challenge review, not a code merge. The user requests a WACV 2027 Vision4Neuro paper derived from the seven top-level notebooks, which are the source of truth. The original supplied PDF was converted to original-draft.txt. Review manuscript.tex and bibliography against notebook text, implementation, OOF artifacts, and evidence audits. Do not treat later RF rerun as the main notebook study. No edits, recursive Codex, MCP, or external tools. Only read files, using nl -ba for exact citations. Ask whether the inference is justified, not merely whether text repeats code. Scope is a bounded empirical pilot, not clinical deployment. For each question give HOLDS, BROKEN or NOT REACHED with quotes: (1) every numerical claim, denominator, weighting, bootstrap and comparison is faithful; (2) Section 4.1 masking, 12 landmarks/all33, coordinate chain and teacher readout are accurate; (3) same-seed runs, source vs participant grouping, train/validation boundaries and notebook vs additional reanalysis are distinguished; (4) citations support scientific claims and do not hallucinate; (5) negative-result wording neither establishes equivalence nor causal temporal benefit and identifies consequential limitations. Name claimed properties for which the retained experiments provide no evidence. Prioritize concrete scientific or numerical defects and missing reproduction details, not generic wish lists for new experiments. Acknowledge known limitations already stated, do not report them as undisclosed. Report line count and last line of manuscript.tex as reading anchor. Each defect requires exact file:line, consequence, fix and severity. End exactly Verdict: approve or Verdict: needs-attention.

## How to review

1. Read with line numbers so every citation is exact: `nl -ba <path>` for
   the new version and `git show 99365f9f96059dcb246dbda1d65151fe38b126a5:<path> | nl -ba` for the old one.
2. For every file the range touches, read the version at `99365f9f96059dcb246dbda1d65151fe38b126a5` as well
   as the new one. Many defects are only visible as a difference: something
   the old code did that the new code silently stopped doing, or a state the
   old code never wrote.
3. For each defect, check whether it already exists at `99365f9f96059dcb246dbda1d65151fe38b126a5`
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
   (`git ls-tree 99365f9f96059dcb246dbda1d65151fe38b126a5 <path>` against `d4b546a22011660e1ddabe7237dd9075dae04eb9`) before saying a binary
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

- It names a location as `path:line` in `d4b546a22011660e1ddabe7237dd9075dae04eb9`, or in `99365f9f96059dcb246dbda1d65151fe38b126a5` when
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
