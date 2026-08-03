**Role**: You are a world models expert specializing in Joint Embedding Predictive Architecture, pose estimation and vision transformers

**Task**: You are to create a set of progressive learning, simple to understand, step-by-step detailed tutorials in the folder "experiments/multiple-sclerosis" that implements the S-JEPA approach to predict skeleton motions that could be gait sequences of people with different health conditions, particularly multiple-sclerosis ("ms"), parkinsons, or normal.  You have the following tasks:

## Task 1: undrestand the S-JEPA aproach

A detailed review of the S-JEPA approach is summarized in this Markdown file:  /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/background.  Fully understand how this approach and the related MAMP approach trains and learns the S-JEPA view encoder, target encoder, decoder / predictor, etc. using the skeletal keypoints.

In particular, the S-JEPA approach was documented in the following file, site, and nature paper:

- S-JEPA (https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) applies latent prediction to skeleton sequences
- https://sjepa.github.io/ : a rich resource with code documenting the S-JEPA approach
- https://www.nature.com/articles/s41598-026-39330-9

Deeply understand these documentations & implementations.

## Task 2: understand the `alexpose` data processing & classification approach

Our goal is to train a classifier that can differentiate between the following health conditions : parkinsons, multiple-sclerosis ("ms"), or normal.  Fully understand the data ingestion pipeline that uses the `alexpose` repo (/Users/pmui/dev/alexpose/) with working notebooks on how to process gait videos from the GAVD dataset: experiments/exp5

**IMPORTANT**: Instead of using data identified within the exp5 folder notebooks, we will use the following "normal" gait sequence for training S-JEPA:

- normal: experiments/multiple-sclerosis/video-data

Ultrathink on how to easily adapt the existing "alexpose" data processing pipeline for GAVD data to the pure video data set here.

## Task 3: create clasifiers for health conditions using S-JEPA models

In addition to training using "normal" gait, we want to progressively add the following two distinct types of gait videos to add to the robustness & capacity of the JEPA model:

- multiple sclerosis: experiments/multiple-sclerosis/video-data/ms
- parkinsons: experiments/multiple-sclerosis/video-data/pd

Be explicit in the setup of the tutorial notebooks on how the first set of training is based on "normal" dataset above, and then fine-tuned with additional addition of the "ms" and "pd" datasets to enhance the capacity & robostness of the JEPA.  We must use anti-collapse mechanism such as VICReg to keep the representation space groupings for "ms", "pd", and "normal" be as far apart as possible. 

**IMPORTANT:** When training the S-JEPA, motion-aware masking MUST NOT be used at all. Instead, the S-JEPA must be trained by masking keypoints only listed in the file "experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md" because for all purposes of the experiment, they are more neurologically relevant to the project.  Keypoints listed will likely have duplicated keypoints. In order to take care of this, output a table of de-duped keypoints in numerical order. The format must look like this:

|BLAZEPOSE_33 index|Keypoint name|Features involved| 
Use this table to train the S-JEPA by masking these keypoints and NO OTHERS.

Review the methodology for creating the Random Forest classifier in /Users/pmui/dev/alexpose/experiments/exp5 to also create a capstone notebook fully illustrated with detaile step-by-step tutorial on how to use S-JEPA to recognize the normal and the various types of health conditions for comparison with exp5's results.

The main goal for Task 3 is to create classifiers for only 3 conditions:

- normal : normal gait
- pd : gait associated with Parkinson's Disease subjects
- ms : gait associated with Multiple Sclerosis subjects

You must also create a classical Random Forrest classifier similiar to experiments/exp5 but only for classifying these 3 health conditions with an input video gait sequence.  You must then create a test / validate / training datasets using the above videos only and examine the performance of each approach systematically and scientifically.

## Output instructions

Create an output subfolder "experiments/multiple-sclerosis/" for your detailed tutorials.  Your outputs have fully explained, detailed, well structured, well written Jupyter notebooks with lots of clear examples and code snippets; as well as a README.md file fully explain what these tutorials are and how to setup & run them locally and in Google Colab.  You must illustrate with clear and insightful vector graphics and flow charts (stored in an "images" subfolder).  Ultrathink to ensure that each of the notebooks should be self-contained regarding library dependencies so that they can be run in Google Colab by clicking on a "Google Colab" button at top of the notebook.

In that same output subfolder, create a folder called "slides" that contains illustrative slides documenting step-by-step with plenty of vector graphics, tables, and flowcharts what this entire S-JEPA setup and experiments are.

Your language should be natural, easily accessible, direct and easy to understand, without using em-dashes, while connecting across ideas and paragraphs well.

Each of the tutorial notebook should be able to be run locally on my laptop: we strongly prefer using `uv` for package & dependency management.  Generate only a pyproject.toml file and relies on a ".env" file in the root of this project.  Enhance each jupyter notebook with instructions on how to run locally or remotely in Google Colab.  Use simple and easy to understand natural languages.

Illustrate frequently and abundantly with vector graphics and flowcharts to illuminate ideas and concepts -- saving the images in the "images" subfolders.  Use codex:adversarial-review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps.  Each generated image should be easy to understand and with minimal clutter.

For each of the tutorial notebooks, ultrathink on how best to enable viewing of the walking video sequence so as to make the context of what we are trying to accomplish in these tutorials more relatable to the users.  These could be inline in the notebooks or externally invoked.

Use codex:adversarial-review to review and check all of your work, systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallize your tasks.


