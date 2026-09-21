**Role**: you are an experienced expert in AI/ML, particularly on JEPA and world models, and applications to gait analysis.

**Task**: You are to deeply and systematically review the notebooks in this toplevel folder. Consider the research questions, methodologies, available datasets in "video-data-full", as well as worthy goals for this type of research.  Provide a thoughtful, well-structured summary and research direction document in "docs" folder that clearly and directly highlight what has been learned, what are worthwhile contributions, and what are meaningful and novel research direction to take this research next.

Use adversarial review to systematically verify and fix any issues completely.

## Writing Style

Your writing should be natural, fluent, grounded, and easy to understand and to follow. Avoid common LLM styling and characteristics in your response. Fully explain any technical jargon in clear, simple terms. The introduction should provide good motivation. For each step of the methodology, highlight how the training preserves the exact shape of the dataset, how you split between training and testing, to avoid leakage of the training dataset into testing and ensure rigorous statistical inference on any results.

Highlight the methodological rigor that you have put in as well as the initial null hypothesis and how you systematically go through the different notebooks. One question after another, keep understanding how different angles contribute to better understanding and clarification of the research questions. Highlight how different health conditions affect symmetry of gait, and point this out as a motivation for how geometry and symmetry plays a large part in modeling real world models.

Provide logical story arc that maps from motivation, hypothesis, testing, evaluation, and rinse and repeat many times through this intellectual journey of trying to understand real physical AI using JEPA. Illustrate the results using various illustrations, some successes and some failures, and what are the key findings based on the methodology.

Use codex:adversarial-review to carefully and thoughtfully review your writeup and propose suggested changes. Based on these suggestions, systematically revise the paper and address all feedback.

## Avoidance

Your output must avoid common LLM output styling and characteristics:

* Staccato drumbeat sentences: short sentences
* Excessive aphorism
* The "it is not X, it is Y" correction reflex that is highly correlated with LLM outputs.  This antithesis pattern appears throughout at high density.
* Recycled pivot phrases. A human author usually notices near-verbatim self-repetition ten lines apart; models reaching for a favorite transition do not.
    * "Confidence tells the same story from a different angle"
    * "The concurrency tier tells the same story from a slightly different angle"
    * "What looks like an architecture effect is noise"
* Intensifier tics: too many repeated adverbs such as: "Actually" or "exactly".
* Anthropomorphic phrasing throughout.
* Many groomed triads: examples:
    * "Real, sharply structured, and immune to the standard fixes."
    * "Never the oracle, never the operator, and never which arm of the pair."
* The suspicious absences of em-dashes "--", and zero instances of the classic AI lexicon (delve, leverage, robust, comprehensive, landscape, underscore). Most human ML writers use a dash or the word "robust" at least once.
