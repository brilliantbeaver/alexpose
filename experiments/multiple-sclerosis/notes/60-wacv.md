**Role**: You are an expert AI/ML researcher specializing in Neural Decoding and Computer Vision for Behavioral and Clinical Neuroscience

**Tasks**: Your goal is to write a highly relevant paper for submission to the WACV 2027 workshop Vision for Biomedical Discovery whose call for paper is at: https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html.

You will then first review a draft paper at "docs/wacv2007/wacv2007-v1.pdf"

Then, fully understand the jupyter notebooks (*.ipynb) and results in them.  You are to correct and refine the draft paper with grounded data, verifiable results, and defensible inferences.  You are to create 7 iteratively improving drafts of the paper; grade each draft with a highly relevant rubric from the conference website, along with critiques as well as suggestions to improve.  Each draft must carefully and thoughtfully consider these suggestions and critiques before creating an enhanced next version.  Output your scores for each draft in the same folder.

Your writing should be natural, fluent, grounded, and easy to understand and to follow.   Avoid common LLM styling and characteristics in your response.

## Avoidance

Your output must avoid common LLM output styling and characteristics:

* Staccato drumbeat sentences: short sentences
* Too many aphorism.
* The "it is not X, it is Y" correction reflex that is highly correlated with LLM outputs.  This antithesis pattern appears throughout at high density.
* Recycled pivot phrases.
* Anthropomorphic phrasing.
* Groomed triads: examples:
    * "Real, sharply structured, and immune to the standard fixes."
    * "Never the oracle, never the operator, and never which arm of the pair."
* The suspicious absences of em-dashes, and zero instances of the classic AI lexicon (delve, leverage, robust, comprehensive, landscape, underscore). Most human ML writers use a dash or the word "robust" at least once.


## Output Instructions

Your document output will be a well-structured, fluent, clear, well-craft documents in the folder: "docs/wacv2007" following the paper preparation outlined in the conference website: https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html

You must verify and check your claims, facts, and inferences to ensure that your response is accurate, correct, useful, and grounded. 

Use Codex adversarial review to review your thinking, inference, results, and writing carefully.  Fix all issues systematically and thoroughly.

Fan out subagents and use dynamic workflows.

---

We notice a few problems in the draft paper: systematically and completely find the root causes and fix these in your drafts:

- Section 4.1 needs to be carefully reviewed for correctness and completeness
- Verify bootstrap: same reported percentile range [-0.121, 0.231] for both RF and mean pose?
- Run a numerical consistency audit with our results
- Fix coordinate description: “normalized (x,y) image coordinates” in Section 3 & “pixel x, pixel y” in 4.2 
- Run a full audit of the sources to ensure that author names, years, titles, etc. is correct and GPT is not hallucinating

---

For any figures included, use best UI/UX skills to ensure that text and lines in these figures are clearly visible without overlaps.  We strongly prefer simple straight lines and symmetrical design.  

---

Your writing should be targetting educated researchers who might not be AI/ML specialists.  Avoid too much AI/ML or statistical jargons or deep technical phrasing.  Instead, use plain professional writing for the general research audience.

---

Further iterate an improved v9 draft of this paper to include:

- The abstract to end with a positive view of the study such as: "Macro-F1, which averages F1 across the three labels, is 0.457 for S-JEPA and 0.452 for mean pose when every source recording receives equal total weight, and 0.411 for RF). On this small-scale dataset, the S-JEPA approach was able to learn meaningful features, with some small performance improvements over a traditional random forest classifier."

- The paper should remove stringent statistical bend that highlights no significant uplift of performance for S-JEPA. 

- The paper should instead highlight that this study has shown real representational learning in a way that S-JEPA's performance after training with just a small sample of videos can beat that of a handcrafted Random Forest Classifier using a handcrafted feature set.

- Overall, the paper should reduce statistical jargons and instead read more fluently for the general technical and research audience.