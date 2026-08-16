**Role**: You are a world models expert specializing in Joint Embedding Predictive Architecture, pose estimation and vision transformers

**Task**: You are to carefully & systematically revise the notebooks (*.ipynb) such that the S-JEPA "View encoder" and "Predictor" were trained first using only "normal" gait.  Only after all the "normal" gait training are done would you progressively add the following distinct types of GAVD gait videos to add to the robustness & capacity of the S-JEPA model:

- parkinsons: /Users/pmui/dev/alexpose/data/gavd/parkinsons
- stroke: /Users/pmui/dev/alexpose/data/gavd/stroke
- myopathic: /Users/pmui/dev/alexpose/data/gavd/myopathic
- cerebralpalsy: /Users/pmui/dev/alexpose/data/gavd/cerebralpalsy

<non-negotiables>
Be explicit in the setup of the tutorial notebooks on how the first set of training is based on "normal" datasets, and then fine-tuned with additional addition of "parkinsons", "stroke", "myopathic", and "cerebralpalsy" datasets to enhance the capacity & robostness of S-JEPA.  We must use anti-collapse mechanism such as VICReg to keep the representation space groupings for "parkinsons", "stroke", "myopathic", "cerebralpalsy", and "normal" be as far apart as possible. 

Explain your reasoning, methodology, process, and result observations clearly and plainly in the notebooks as well as in the related "docs" and README.md files.  Use natural and simple languages in an easily accessible and compelling way.
</non-negotiables>


**IMPORTANT:** When training the S-JEPA, motion-aware masking MUST NOT be used at all. Instead, the S-JEPA must be trained by masking keypoints only listed in the file "experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md" because for all purposes of the experiment, they are more neurologically relevant to the project.  Keypoints listed will likely have duplicated keypoints. In order to take care of this, output a table of de-duped keypoints in numerical order. The format must look like this:

|BLAZEPOSE_33 index|Keypoint name|Features involved| 
Use this table to train the S-JEPA by masking these keypoints and NO OTHERS.
