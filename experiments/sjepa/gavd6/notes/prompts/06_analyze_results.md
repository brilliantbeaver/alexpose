**Role**: You are an expert AI/ML researcher specializing in world models and JEPA.

**Task**: Based on the latest methodology and results in `docs/studies/synthetic-training-v2`, fully and systematically brainstorm and draft a paper title and abstract using a thoughtful and appropriate multi-dimensional scoring system to guide your reasoning, along with specific critiques as well as concrete suggestions on how to improve. The goal is to iterate the abstract and make it strongly relevant and compelling to the ICLR main track. Scope the paper around whether paired synthetic supervision can improve 2D pose trajectories while preserving movement amplitude, timing, and left–right joint relationships. Build the argument around the findings from the laterality study and the original synthetic training study.

* https://iclr.cc/

Iteratively create 7 versions of the abstract without overriding existing files by thoughtfully incorporating each of the successive versions' critique and suggestions. Each version must improve upon previous versions by carefully reviewing and selecting the most appropriate results, inferences, and findings from the results, as well as relevant literature from authoritative primary sources such as research papers and their official implementations. Distinguish real-data results, results available only as summaries, synthetic demonstrations, and proposed experiments. Be thoughtful and strategic in what to select and what to ignore in order to create an ever more compelling, cogent, and intellectually relevant paper abstract. This is a manuscript revision using existing evidence; recompute supported analyses where necessary, label new analyses as exploratory, and describe experiments without completed evidence as future work.

Use the same scoring rubric throughout: relevance and contribution to the conference (20%), claim accuracy and evidence support (20%), evaluation and statistical rigor (15%), scientific insight and positioning against related work (15%), reproducibility (10%), clarity and narrative (10%), figures (5%), and submission fit (5%). Score each dimension from 0 to 10 and report the weighted total out of 100, with concrete reasons and remaining weaknesses. A score of 5 indicates a substantial unresolved weakness and 10 indicates no material weakness within the stated scope. Distinguish improvements achievable through revision from limitations that require new data or experiments; a later version should not receive a higher score merely because it is later. To illustrate the data processing pipeline, thoughtfully create a modern, clean, and simple illustration using vector graphics of the training pipeline. 

Provide 7 suggestions of relevant titles that reflect the objective of the paper. The abstract should strongly reflect the main theme of the conference. Save your scores, suggestions, and critiques of each revision in `docs/studies/synthetic-training-v2`, preserving the existing record. Each revision of this paper should be saved in `docs/studies/synthetic-training-v2` with a thoughtfully useful filename and version number.

## Writing Style

Your writing should be natural, fluent, grounded, and easy to understand and to follow. Avoid common LLM styling and characteristics in your response. Fully explain any technical jargon in clear, simple terms.

Your output must avoid common LLM output styling and characteristics:

* Staccato drumbeat sentences: short sentences
* Humans land an aphorism occasionally; LLMs land one every time, and they close nearly every section and the abstract this way.
* The "it is not X, it is Y" correction reflex that is highly correlated with LLM outputs.  This antithesis pattern appears throughout at high density.
* Recycled pivot phrases. A human author usually notices near-verbatim self-repetition ten lines apart; models reaching for a favorite transition do not.
    * "Confidence tells the same story from a different angle"
    * "The concurrency tier tells the same story from a slightly different angle"
    * "What looks like an architecture effect is noise"
* Intensifier tics. "Actually" appears 20 times, "exactly" 8 times: That is a lot of emphasis with no technical context.
* Anthropomorphic phrasing throughout.
* Groomed triads are dotted throughout: examples:
    * "Real, sharply structured, and immune to the standard fixes."
    * "Never the oracle, never the operator, and never which arm of the pair."
* The suspicious absences. Zero em-dashes at all, and zero instances of the classic AI lexicon (delve, leverage, robust, comprehensive, landscape, underscore). Most human ML writers use a dash or the word "robust" at least once.
