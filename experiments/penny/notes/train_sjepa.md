**Role**: You are a world models expert specializing in Joint Embedding Predictive Architecture, pose estimation and vision transformers

**Task**: You are to create a set of progressive learning, simple to understand, step-by-step detailed tutorials in the folder "gait/skeleton-jepa/gavd4-sjepa" that implements the S-JEPA approach to predict skeleton motions that could be gait sequences of various health conditions.  You have the following tasks:

## Task 1: undrestand the S-JEPA aproach

A detailed review of the S-JEPA approach is summarized in this Markdown file:  /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/background.  Fully understand how this approach and the related MAMP approach trains and learns the S-JEPA view encoder, target encoder, decoder / predictor, etc. using the skeletal keypoints.

In particular, the S-JEPA approach was documented in the following file, site, and nature paper:

- S-JEPA (https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) applies latent prediction to skeleton sequences
- https://sjepa.github.io/ : a rich resource with code documenting the S-JEPA approach
- https://www.nature.com/articles/s41598-026-39330-9

Deeply understand these documentations & implementations.

## Task 2: understand the `alexpose` GAVD data processing & classification approach

Our goal is to train a classifier that can differentiate between the following health conditions using the GAVD dataset and ingestion pipeline that uses the `alexpose` repo (/Users/pmui/dev/alexpose/) with working notebooks on how to process gait videos from the GAVD dataset:

/Users/pmui/dev/alexpose/experiments/exp5

**IMPORTANT**: Instead of using data identified within the exp5 folder notebooks, we will use the following "normal" gait sequence folder with GAVD csv files for training S-JEPA:

- normal: /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/data-gavd/normal

Ultrathink on how to easily enable each of your tutorials to use real YouTube downloaded data based on the CSV files in the "normal" subfolder.

For saving the download for each of the unique GAVD youtube videos, we will store them in the "normal" subfolder of the following folder (if the videos have not been downloaded already):

/Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/youtube


## Task 3: create clasifiers for health conditions using S-JEPA models

After training S-JEPA on what are "normal" gait sequences, we will use the following folders' csv files to create classifiers to identify what are health conditions that are different than "normal":

- parkinsons: /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/data-gavd/parkinsons
- stroke: /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/data-gavd/stroke
- cerebralpalsy: /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/data-gavd/cerebralpalsy
- myopathic: /Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/data-gavd/myopathic

Ultrathink on how to easily enable each of your tutorials to use real YouTube downloaded data based on the CSV files in the above health conditions subfolders.

For saving the download for each of the unique GAVD youtube videos, we will store them in the corresponding conditions subfolder of the following folder (if the videos have not been downloaded already):

/Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4/youtube

**IMPORTANT:** When training the S-JEPA, motion-aware masking must not be used at all. Instead, the S-JEPA must be trained by masking keypoints only listed in the following files because for all purposes of the experiment, they are more neurologically relevant to the project: 
    "PD_keypoint_mapping.md",
    "CP_keypoint_mapping.md",
    "MYO_keypoint_mapping.md", and
    "STROKE_keypoint_mapping.md"

However, the keypoints listed in these four will likely have many duplicated keypoints. In order to take care of this, output a table of de-duped keypoints in numerical order. The format must look like this:
|BLAZEPOSE_33 index|Keypoint name|Features involved| 
Use this table to train the S-JEPA by masking these keypoints and NO OTHERS.

Review the methodology for creating the Random Forest classifier in /Users/pmui/dev/alexpose/experiments/exp5 to also create a capstone notebook fully illustrated with detaile step-by-step tutorial on how to use S-JEPA to recognize the normal and the various types of health conditions for comparison with exp5's results.


## Output instructions

Create an output subfolder "gait/penny/gavd3" for your tutorials.  Your outputs have fully explained, detailed, well structured, well written Jupyter notebooks; as well as a README.md file fully explain what these tutorials are and how to setup & run them locally and in Google Colab.  You must illustrate with clear and insightful vector graphics and flow charts (stored in an "images" subfolder).  Ultrathink to ensure that each of the notebooks should be self-contained regarding library dependencies so that they can be run in Google Colab by clicking on a "Google Colab" button at top of the notebook.

In that same output subfolder, create a folder called "slides" that contains illustrative slides documenting step-by-step with plenty of vector graphics, tables, and flowcharts what this entire S-JEPA setup and experiments are.

Your language should be natural, easily accessible, direct and easy to understand, without using em-dashes, while connecting across ideas and paragraphs well.

Each of the tutorial notebook should be able to be run locally on my laptop: we strongly prefer using `uv` for package & dependency management.  Generate only a pyproject.toml file and relies on a ".env" file in the root of this project.  Enhance each jupyter notebook with instructions on how to run locally or remotely in Google Colab.  Use simple and easy to understand natural languages.

Illustrate frequently and abundantly with vector graphics and flowcharts to illuminate ideas and concepts -- saving the images in the "images" subfolders.  Use codex:adversarial-review of your generated vector graphics and workflow to ensure that they are not cluttered, and that no excessive text or line overlaps.  Each generated image should be easy to understand and with minimal clutter.

For each of the tutorial notebooks, ultrathink on how best to enable viewing of the walking video sequence so as to make the context of what we are trying to accomplish in these tutorials more relatable to the users.  These could be inline in the notebooks or externally invoked.

Use codex:adversarial-review to review and check all of your work, systematically and thoughtfully fix all issues.

Use fan out subagents with dynamic workflows to parallize your tasks.


