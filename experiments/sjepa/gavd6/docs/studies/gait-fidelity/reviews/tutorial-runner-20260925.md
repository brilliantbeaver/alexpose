# Tutorial execution runner review — September 25, 2026

The execution runners preserve the source notebooks and retain populated copies
separately. They use temporary kernelspecs bound to the invoking interpreter,
so a different user-registered `python3` kernel cannot silently change the
validation environment.

The independent runner review identified two evidence-recording defects:

1. A failed rerun could leave a previous successful `execution.json` untouched.
2. Computing a source hash after execution could identify edited text rather
   than the notebook bytes actually parsed and executed.

Both runners now replace previous success receipts with a running state before
starting work and record failures or interruptions. The core tutorial runner
retains the completed notebook records, failing notebook, stage, original source
hash and exception details. The walkthrough runner records the failing stage
and exception. Failure-receipt write errors preserve the original exception.
Success receipt fields retain their previous meanings and names.

Each runner reads source bytes once, computes their SHA-256 hash, and parses
those same bytes. The walkthrough's optional export switch and both runners'
Markdown link relocation still modify only the populated copy. The recorded
hash identifies the original cleared source notebook, before those declared
output transformations.

The core runner rejects work/output paths inside the source notebook directory
and output paths inside the fixture work directory. The latter would otherwise
make a new work directory nonempty before fixture initialization. The walkthrough
retains its guards against output inside source notebooks or the evidence packet.
The shared `configure` helper rejects follow-up sessions before constructing core
tutorial caches or launching a core workflow.

## Targeted validation

Fault-injection tests exercised runner control flow with mocked clients. They
did **not** launch Jupyter kernels, fit models, or establish notebook numerical
correctness; genuine kernel execution and mathematical parity checks are
recorded separately.

- A simulated second-notebook failure replaced an earlier core-run pass with a
  failed receipt, retaining the first completed record and identifying the
  failing notebook and exact source hash.
- A simulated fixture initialization failure replaced a previous pass before
  notebook execution began.
- Successful core-run receipts retained their expected fields and source hashes.
- All three core-run path guards rejected unsafe/conflicting output locations.
- A simulated walkthrough failure and keyboard interruption produced the correct
  failed/interrupted states and restored the prior `JUPYTER_PATH`.
- A successful mocked walkthrough preserved the exact existing `PASS` field set,
  export flag and destination, figure/error counts, and relocated analysis link.
- The source notebooks and all 71 files then present in `outputs/iclr` had
  identical SHA-256 hashes before and after these tests. No source training
  arrays or checkpoints were opened or modified.

All temporary test files were confined to automatically removed temporary
directories. The current relative Markdown links contain no fragments or titles;
their relocated targets were checked against the original lesson/doc locations.
