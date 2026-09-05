# Figure status

The existing PDF and SVG figures in this directory belong to the superseded exploratory, transductive analysis. They are retained only as historical working assets. They are not figures for the laterality v2.1 protocol and must not be included in a v2.1 manuscript, presentation, or result summary.

Several details make the old assets incompatible with the current study. They describe a six-pair temporal-variation target rather than the registered five-pair, paired-valid velocity target. They also describe one encoder that had seen the evaluated sequences, repeated reshuffled cross-validation, legacy cohort counts, and legacy performance estimates. The v2.1 design instead trains a fresh encoder for each of five outer source-video folds, uses four inner folds only for read-out selection, and reserves each outer-test source from representation training and model selection.

The current Markdown and LaTeX manuscripts intentionally do not reference these figures. New figures should be generated only from complete v2.1 held-out evaluation artifacts carrying the matching protocol, cohort, split, checkpoint, and implementation digests. Until then, conceptual diagrams must also be checked against the registered five-pair target and source-level split before use.
