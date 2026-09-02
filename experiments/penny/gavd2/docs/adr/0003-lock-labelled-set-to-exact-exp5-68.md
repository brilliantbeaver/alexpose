# ADR 0003: Lock the labelled probe set to the exact exp5 68 by sequence id

**Status:** accepted

## Context

iteration 1 marked the labelled subset by taking the first N sequences of each of the
five classes. That is close to, but not identical with, the 68 sequences the prior
Random Forest actually trained on. It provably diverges for cerebralpalsy (the full GAVD
tree has more cerebral-palsy sequences than exp5 curated) and myopathic, swapping in
sequences the baseline never saw. A comparison on a different labelled set is not
controlled.

There is also a spelling trap: the full GAVD tree names the folder `cerebral palsy`
(with a space), while the exp5 curated tree and its feature pickle use `cerebralpalsy`
(no space). If the label is canonicalized in one place but not another, the cerebral-
palsy class silently drops to zero.

## Decision

Resolve the exact 68 exp5 sequence ids and mark only those as labelled. Use a three-tier
resolver in notebook 00: (1) unpickle the exp5 82-feature file and read each object's
`sample_id` and `condition_label` off the `all_features` list; that unpickle needs the
private ambient package, so in the notebook's own environment it fails and we fall
through; (2) glob the five curated exp5 CSV folders, whose file names are the sequence
ids; (3) a checked-in `CANONICAL_68` constant for a bare Colab. A real locked run
fail-stops if all three fail rather than silently using first-N; an explicit exploratory
mode can use first-N but writes to a separate cache namespace. The condition label is
canonicalized to `cerebralpalsy` unconditionally everywhere, and a separate map recovers
the on-disk folder spelling (`cerebral palsy`) when reading files.

Verified fact: the 68 exp5 ids equal the GAVD `seq` column values exactly and each
exists in the full tree, so the lock is derivable. The exp5 pickle is a dict
`{all_features: [68 feature objects], condition_counts, metadata}`, not a
`{sample_id: label}` dict; the resolver reads the list, not a flat dict.

## Consequences

- The probe evaluates on exactly the sequences the baseline used (subject to download
  and extraction coverage, reported as "N of 68").
- Every cache artifact is stamped with a canonical id hash so a stale-artifact mix is
  caught.
- The canonicalization must be threaded through notebooks 00, 02, 03, and 05; a miss in
  any of them drops the cerebral-palsy class. This is called out in each notebook.
