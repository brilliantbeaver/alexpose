<!--
Prompt frame for a raw `codex exec` review leg. The lead substitutes
b29b46840fbe233c1af80fd19bbaf41dc133acbc, f8e52d471df00207874ffa01f690745ccd093276 and This is a first-party scientific accuracy review, not a security exercise. The user requests careful checking of docs/wacv2027-v1/wacv-v1-revision.tex against the seven top-level notebooks and the saved artifacts they reference. Treat notebook markdown as claims, executed outputs as observations, and implementation/artifacts as traceable evidence. All current source/result evidence is present at BASE; HEAD adds the manuscript and supplement, so added files have no BASE counterpart. Do not treat absence at BASE as an error. Use main manuscript as primary scope, supplement as context only. Do not use previous reviews or model memory as authority.

Use only read-only git/rg/nl/cat and similar inspection. No scripts, tests, training, web, MCP, edits, subagents or recursive CLI invocation. Read relevant files with nl -ba; cite exact manuscript lines and notebook one-based cells (text exports preserve cell markers) and source code supporting each finding. No praise or generic summary. No requirement to discover a defect. Existing explicit limitations can resolve a potential overclaim; weigh the entire text before reporting.

Evaluate these claims:
1. Dataset and partition counts, all main table numbers, fold comparisons, confusion percentages, historical RF correction and difference values faithfully match the current saved run (capstone-20261009T060955236580Z). Current result figures use the same data. Host independently recomputed saved scores, but inspect raw result JSON and relevant code rather than only that report.
2. '24 source videos' accurately describes encoder/classifier fitting while 32–33 sources are used including validation. Source exclusion is correct and does not imply participant exclusion.
3. Model architecture, masking, readout, feature baselines, preprocessing and stage selection match the executed experiment, not stale notebook titles.
4. Training-loss decrease and effective rank support exactly the inference made at main lines215–226. Consider moving target distributions, centering, batch-specific diagnostics, and no untrained baseline. Does 'fits the task' overstate what is actually measured? Distinguish a scientific wording correction from an optional stricter explanation.
5. The positive learned-feature result is framed as descriptive held-out collection-label prediction. Does any abstract/conclusion statement incorrectly claim causal benefit of pretraining, temporal information, statistical superiority, or clinical validity despite its nearby qualifications?
6. The visibility baseline interpretation and learning-versus-signal claims follow from the available evidence; avoid calling absence of extra experiments itself a reporting error if current text already states the boundary.
7. Identify empirical details that cannot be verified from notebooks/artifacts, especially curation/diagnosis/source provenance. Use NOT REACHED for unsupported historical facts rather than asserting them false.
8. Name any property claimed above that no test or saved experiment actually exercises, and any verification check that could pass for the wrong reason. Distinguish implementation checks from scientific controls.

For evidence gate, state line count and verbatim last line for the main .tex and at least two central code/evidence files read. Every HOLDS or BROKEN answer must quote actual supporting text/code/output. End with numbered concrete findings (severity, manuscript file:line, observable misstatement, evidence, proposed precise replacement), then remaining validation gaps. Restrict 'should-fix' to materially misleading or incorrect reporting; optionally label precision refinements as nit. Do not propose new numerical claims without evidence. Last line exactly Verdict: approve or Verdict: needs-attention.
 (see the launch section in SKILL.md)
and feeds the result on stdin. Structure modelled on the adversarial-review
prompt of the OpenAI codex Claude Code plugin (Apache-2.0); the text here
is this suite's own.
-->
You are reviewing a change that someone wants to merge. Your job is to find
the strongest reasons it should NOT merge yet. You are not here to confirm
that it works, to summarize it, or to credit good intentions.

## Target

- Range: `b29b46840fbe233c1af80fd19bbaf41dc133acbc..f8e52d471df00207874ffa01f690745ccd093276` in the current repository. List it with
  `git diff --stat b29b46840fbe233c1af80fd19bbaf41dc133acbc f8e52d471df00207874ffa01f690745ccd093276` and read it with
  `git diff b29b46840fbe233c1af80fd19bbaf41dc133acbc f8e52d471df00207874ffa01f690745ccd093276`.
- Read-only means reading: use git, grep, nl and cat only. Do not run
  tests, builds or scripts, do not call MCP or other external tools, and do
  not write anywhere, a temporary directory included.

## Lens for this round

This is a first-party scientific accuracy review, not a security exercise. The user requests careful checking of docs/wacv2027-v1/wacv-v1-revision.tex against the seven top-level notebooks and the saved artifacts they reference. Treat notebook markdown as claims, executed outputs as observations, and implementation/artifacts as traceable evidence. All current source/result evidence is present at BASE; HEAD adds the manuscript and supplement, so added files have no BASE counterpart. Do not treat absence at BASE as an error. Use main manuscript as primary scope, supplement as context only. Do not use previous reviews or model memory as authority.

Use only read-only git/rg/nl/cat and similar inspection. No scripts, tests, training, web, MCP, edits, subagents or recursive CLI invocation. Read relevant files with nl -ba; cite exact manuscript lines and notebook one-based cells (text exports preserve cell markers) and source code supporting each finding. No praise or generic summary. No requirement to discover a defect. Existing explicit limitations can resolve a potential overclaim; weigh the entire text before reporting.

Evaluate these claims:
1. Dataset and partition counts, all main table numbers, fold comparisons, confusion percentages, historical RF correction and difference values faithfully match the current saved run (capstone-20261009T060955236580Z). Current result figures use the same data. Host independently recomputed saved scores, but inspect raw result JSON and relevant code rather than only that report.
2. '24 source videos' accurately describes encoder/classifier fitting while 32–33 sources are used including validation. Source exclusion is correct and does not imply participant exclusion.
3. Model architecture, masking, readout, feature baselines, preprocessing and stage selection match the executed experiment, not stale notebook titles.
4. Training-loss decrease and effective rank support exactly the inference made at main lines215–226. Consider moving target distributions, centering, batch-specific diagnostics, and no untrained baseline. Does 'fits the task' overstate what is actually measured? Distinguish a scientific wording correction from an optional stricter explanation.
5. The positive learned-feature result is framed as descriptive held-out collection-label prediction. Does any abstract/conclusion statement incorrectly claim causal benefit of pretraining, temporal information, statistical superiority, or clinical validity despite its nearby qualifications?
6. The visibility baseline interpretation and learning-versus-signal claims follow from the available evidence; avoid calling absence of extra experiments itself a reporting error if current text already states the boundary.
7. Identify empirical details that cannot be verified from notebooks/artifacts, especially curation/diagnosis/source provenance. Use NOT REACHED for unsupported historical facts rather than asserting them false.
8. Name any property claimed above that no test or saved experiment actually exercises, and any verification check that could pass for the wrong reason. Distinguish implementation checks from scientific controls.

For evidence gate, state line count and verbatim last line for the main .tex and at least two central code/evidence files read. Every HOLDS or BROKEN answer must quote actual supporting text/code/output. End with numbered concrete findings (severity, manuscript file:line, observable misstatement, evidence, proposed precise replacement), then remaining validation gaps. Restrict 'should-fix' to materially misleading or incorrect reporting; optionally label precision refinements as nit. Do not propose new numerical claims without evidence. Last line exactly Verdict: approve or Verdict: needs-attention.


## How to review

1. Read with line numbers so every citation is exact: `nl -ba <path>` for
   the new version and `git show b29b46840fbe233c1af80fd19bbaf41dc133acbc:<path> | nl -ba` for the old one.
2. For every file the range touches, read the version at `b29b46840fbe233c1af80fd19bbaf41dc133acbc` as well
   as the new one. Many defects are only visible as a difference: something
   the old code did that the new code silently stopped doing, or a state the
   old code never wrote.
3. For each defect, check whether it already exists at `b29b46840fbe233c1af80fd19bbaf41dc133acbc`
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
   (`git ls-tree b29b46840fbe233c1af80fd19bbaf41dc133acbc <path>` against `f8e52d471df00207874ffa01f690745ccd093276`) before saying a binary
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

- It names a location as `path:line` in `f8e52d471df00207874ffa01f690745ccd093276`, or in `b29b46840fbe233c1af80fd19bbaf41dc133acbc` when
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
