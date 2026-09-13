**Role**: You are an expert AI/ML researcher, technical communicator, and science educator.

**Task**: You are to thoroughly revise the explanations and formatting in your documents for "latent-laterality" in the "docs" and "notes" folders to be much more clear, understandable, grounded, and research-style.

For `proposal.md`, reframe the document as a clear and detailed pitch of the latent laterality idea to a general audience. Do not make any assumptions, and explain everything from first principles. You should aim to use natural language as much as possible, and your writing should use stylistic techniques from ICLR/ICML/NeurIPS papers. This is not a full paper, but it should sound like an abstract or introduction from one of these top-tier conferences.

For `runbook.md`, first change the title to something more research-focused, like "experiments.md". This means that you should shift the document from being a technical handbook to being a research experiment guide to help the reader navigate through the various experiments that will be run in this study. You should make the experiments the main focus of this document, while also clearly and systematically walking the reader through how to set up the experiments from where they currently are. For each step, make sure to explain why you are suggesting a particular change, not just what needs to be different. Use easy to read tables, lists, and sections to allow the user to smoothly navigate this guide.

For `theory.md`, like the `proposal.md`revision, start from first principles instead of jumping straight into how different concepts are applied to the research. You should explain in detail how and why key topics in AI and math are essential to understanding the latent laterality study. Your explanation should be mostly text, like in well-annotated university lecture notes. Make sure to use proper LaTeX rendering.

For concepts that are useful to understanding the research but not directly related to the core experiments, make a organized, systematic list of these topics. Make sure to only include concepts that are real topics in AI or math in and of themselves, rather than surface level descriptions or qualitative analysis. The goal of this document is to allow the reader to gain a real technical foundation in the research.

For compiling your list of topics, focus on topics that would be found in an AI or math class at Stanford. Ultrathink on how to find high quality lectures and notes for Stanford courses and provide links to these resources. If you can not find a high-quality resources for a particular topic in the Stanford archive, search for resources from other top universities such as MIT and CMU. If certain concepts require a deep knowledge of implementation in addition to theory, consult the UvA Deep Learning notebooks, Sebastian Raschka or Andrej Karpathy's Youtube videos, or any other high quality resource that you can find.

For `swap-probe.md`, follow the same revision instructions as `runbook.md`. Your output should be detailed, well structure, and well written.

For each of your document revisions, you must illustrate with clear and insightful vector graphics and flow charts (stored in an "images" folder). You will use these illustrations to illuminate ideas and concepts. Use adversarial review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps. Each generated image should be easy to understand and with minimal clutter.

After each of your revisions, carefully consider how you could make the revision even better and clearer with concrete and specific suggestions. Based on those suggestions, thoroughly and systematically revise the documents to a final version that greatly improves the readability and conceptual clarity of the writing. Do not use em-dashes.

Use codex:adversarial-review to review and check all of your work, and systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallelize your tasks.
