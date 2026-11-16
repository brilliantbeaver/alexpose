# Role

You are a world models expert specializing in Joint Embedding Predictive Architecture, pose estimation and vision transformers

# Tasks

You are to create a set of progressive learning, simple to understand, step-by-step detailed tutorials in the folder "experiments/aaai" that implements the S-JEPA approach to predict skeleton motions that could be gait sequences of people with different health conditions with the goal for detecting what are not "normal" gait.  

**Important**: the goal for this set of experiments and explorations is to write a successful paper for the 4th Workshop on Automated Spatial and Temporal Anomaly Detection (ASTAD) which has a due date in a few days:  https://ismart.ece.mcgill.ca/astad4/

Our research in this project should align with AI methods for detecting, predicting, explaining, and responding to anomalies, abnormalities, outliers, risks, failures, and deviations from expected or learned normal behavior across spatial, temporal, and multimodal data. 

You have the following tasks:

## Task 1: undrestand the S-JEPA approach with stochastic positional embeddings

A detailed review of the S-JEPA approach is summarized in this Markdown file:  /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/background.  Fully understand how this approach and the related MAMP approach trains and learns the S-JEPA view encoder, target encoder, decoder / predictor, etc. using the skeletal keypoints.

In particular, the S-JEPA approach was documented in the following file, site, and nature paper:

- https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf : S-JEPA applies latent prediction to skeleton sequences
- https://sjepa.github.io/ : a rich resource with code documenting the S-JEPA approach
- https://arxiv.org/pdf/2308.07092 : MAMP stands for Masked Motion Prediction, a self-supervised pre-training framework
- https://openreview.net/pdf?id=jLnygpRFYm : Masked Image Modeling (MIM); Predicting masked tokens in stochastic locations improves masked image modeling; Stochastic Positional Embeddings
- https://www.alphaxiv.org/pdf/2611.09302 : stochasticity (with Stochastic Latent Variables) adds capabilities, error-correlated predictive uncertainty and diverse completions, at no cost to representation quality
- https://www.nature.com/articles/s41598-026-39330-9
- https://www.thecprogramminglanguage.com/jepa.html : JEPA: Joint-Embedding Predictive Architectures

Deeply understand these documentations & implementations.

## Task 2: understand the `alexpose` data processing & classification approach

Deeply understand how to use the python packages in "ambient", with example usage in the folder "experiments/multiple-sclerosis" and its notebooks.

Our goal is to train a classifier that can differentiate between gaits that are normal versus abnormal.  Ultrathink if we should restrict abnormal gaits to only these: parkinsons, stroke, cerebral palsy, myopathic, stroke.  These are related to neurological conditions that we know the etiology and neuroscience of.

**GAVD Datasets for processing**: This project and notebooks should try to use the GAVD dataset as much as possible to satisfy reviewers' demand for reasonably sufficient dataset sizes.  Ultrathink if the GAVD dataset satisfy the generalizability requirements -- and document clearly in the README.md file in the output folder.  The GAVD dataset has been classified in "data/gavd" and deeply consider using the following condition subfolders:

* normal
* parkinsons
* stroke
* cerebral palsy
* myopathic
* stroke

All video downloads must be made to "data/videos-gavd" with thoughtful consideration on folder organizations.

Fully understand the data ingestion pipeline that uses the `alexpose` repo (/Users/pmui/dev/alexpose/) with working notebooks on how to process gait videos from the GAVD dataset here.


## Task 3: create clasifiers for health conditions using S-JEPA models

Training of the JEPA encoders should first start with "normal" gait videos.

Then, we will try to add datasets for not "normal" conditions, including:

* parkinsons
* stroke
* cerebral palsy
* myopathic
* stroke

Be explicit in the setup of the tutorial notebooks on how the first set of training is based on "normal" dataset above, and then fine-tuned with additional addition of additional datasets to enhance the capacity & robostness of the JEPA.  We must use anti-collapse mechanism such as VICReg to keep the representation space groupings as far apart from each other as possible.

**IMPORTANT:** During training, S-JEPA must be trained by using stochastic masking (MAMP), where masking is conducted randomly to select subsets of skeletal keypoints (or joints) across frames uniformly without considering their motion intensity or velocity. Ultrathink when and how best to use Stochastic Latent Variables, and Stochastic Positional Embeddings.  Keypoints listed will likely have duplicated keypoints. In order to take care of this, output a table of de-duped keypoints in numerical order. The format must look like this:

|BLAZEPOSE_33 index|Keypoint name|Features involved|

Use this table to train the S-JEPA by masking these keypoints and NO OTHERS.

Review the methodology for creating the Random Forest classifier in /Users/pmui/dev/alexpose/experiments/exp5 to also create a capstone notebook fully illustrated with detaile step-by-step tutorial on how to use S-JEPA to recognize the normal and the various types of health conditions for comparison with exp5's results.

The main goal for this task is to create robust classifiers using the S-JEPA model outputs for these conditions:

* normal
* parkinsons
* stroke
* cerebral palsy
* myopathic
* stroke

For the set of all GAVD datasets, use the refreshed and updated data in "data/gavd" and not any cached version.  The latest GAVD manifest and dataset count should be at: "data/gavd/README.md".

For performance comparison, you must also create a classical Random Forest classifier similiar to experiments/exp5 but only for classifying these health conditions with an input video gait sequence.  You must then create a test / validate / training datasets using the above videos only and examine the performance of each approach systematically and scientifically.

## Output instructions

Create an output subfolder "experiments/aaai/" for your detailed tutorials.  Your outputs have fully explained, detailed, well structured, well written Jupyter notebooks with lots of clear examples and code snippets; as well as a README.md file fully explain what these tutorials are and how to setup & run them locally.  You must illustrate your notebooks, README, and docs with clear and insightful vector graphics and flow charts (stored in an "images" subfolder).  Ultrathink to ensure that each notebook should be self-contained regarding library dependencies so that they can be run independently

All configuration and API keys are saved into ".env" in the output folder, and will be loaded using python_dotenv library with: "load_env(find_env())" across the notebooks.

In that same output subfolder, create a folder called "slides" that contains illustrative slides documenting step-by-step with plenty of vector graphics, tables, and flowcharts explaining the setup and experiments.

Your language should be natural, full-sentences, easily accessible, direct and easy to understand, without using em-dashes, while connecting across ideas and paragraphs well.

Each of the tutorial notebook should be able to be run locally on my laptop: we strongly prefer using `uv` for package & dependency management.  Generate only a pyproject.toml file and relies on a ".env" file in the root of this project.  Enhance each jupyter notebook with instructions on how to run locally.  Use simple and easy to understand natural languages.

Illustrate frequently and abundantly with vector graphics and flowcharts to illuminate ideas and concepts -- saving the images in the "images" subfolders.  Use adversarial-review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps.  Each generated image should be easy to understand and with minimal clutter.

For each of the tutorial notebooks, ultrathink on how best to enable viewing of the walking video sequence so as to make the context of what we are trying to accomplish in these tutorials more relatable to the users.  These could be inline in the notebooks or externally invoked.

Use critical adversarial-review to check and verify all of your work; systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallize your tasks.



