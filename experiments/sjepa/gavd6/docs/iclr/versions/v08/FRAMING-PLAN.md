# In-place v08 framing revision

The paper should connect the empirical question to self-supervised learning and
JEPA without changing the supervision actually used. JEPA predicts representations
of a target from a context. Here the target encoder receives privileged projected
reference poses and the readout receives coordinate targets. The completed system
is therefore a supervised synthetic adaptation of an idea from self-supervised
learning. It is a study of measurement fidelity in representations that might
support world models, not an evaluated action-conditioned world model.

Geometry motivates the measurement: the knee angle is defined by three named
landmarks; bilateral comparison is meaningful only if anatomical identity and
trajectory shape survive restoration. Clinical stroke, Parkinson's disease, and
osteoarthritis studies motivate preserving side-dependent variation, without
making asymmetry a universal disease marker or turning a synthetic edit into a
clinical condition. Physical mirroring and mistaken observation labels remain
distinct operations. No untested equivariance is asserted.

The narrative will follow the completed questions: whether masked reference-feature
learning improves response, whether pairing feature differences adds value,
whether the downstream objective damages trajectories, and whether geometric
assignment agrees with response. Null contrasts will be explicitly formalized as
an explanatory description of recorded comparisons, not retroactive preregistration.
Notebooks document this progression; unexecuted recipe arms and tutorial fits
remain outside the evidence.

Data integrity needs three separate explanations: source-person split inheritance,
fixed tensor and anatomical/time indexing, and training-only fitting/calibration.
Keeping tensor shape does not leave raw motion unchanged: resampling, synthetic
editing, rendering, and observation corruption are part of the experiment. A reused
development set cannot be promoted to an independent test set by prose or bootstrap.

All prior numerical results remain bound to their artifacts. A new appendix
illustration will display successes and failures across all 14 people using saved
paired errors, without generating pose examples. The paper stays within the
nine-page initial-submission limit; fuller notebook and leakage-control details
belong in the appendix. Prior v08 files are preserved in the history snapshot.

The requested codex:adversarial-review runs after drafting, followed by revisions
and an explicit disposition record. The user's instruction to address feedback
authorizes those revisions; review-only plugin defaults do not cancel that task.
