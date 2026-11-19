# Deduplicated joint whitelist

The selected features use the following 16 distinct BlazePose joints. The table is in numerical order. The model receives only these joints, so its uniform masking sampler cannot mask any other landmark. Original BlazePose indices and compact model positions are different: compact position 0 means BlazePose joint 11.

|BLAZEPOSE_33 index|Keypoint name|Features involved|
|---|---|---|
|11|Left shoulder|Left hip angle; torso scale and orientation; learned shoulder trajectory|
|12|Right shoulder|Right hip angle; torso scale and orientation; learned shoulder trajectory|
|13|Left elbow|Learned arm trajectory and shoulder-elbow-wrist relationship|
|14|Right elbow|Learned arm trajectory and shoulder-elbow-wrist relationship|
|15|Left wrist|Wrist speed, variability and excursion; learned arm swing|
|16|Right wrist|Wrist speed, variability and excursion; learned arm swing|
|23|Left hip|Pelvis center; torso scale; left hip and knee angles; bilateral angle asymmetry|
|24|Right hip|Pelvis center; torso scale; right hip and knee angles; bilateral angle asymmetry|
|25|Left knee|Left hip, knee and ankle angles; bilateral angle asymmetry; learned leg trajectory|
|26|Right knee|Right hip, knee and ankle angles; bilateral angle asymmetry; learned leg trajectory|
|27|Left ankle|Left knee and ankle angles; speed and excursion; ankle separation|
|28|Right ankle|Right knee and ankle angles; speed and excursion; ankle separation|
|29|Left heel|Learned foot orientation and heel trajectory|
|30|Right heel|Learned foot orientation and heel trajectory|
|31|Left foot index|Left ankle angle; learned foot orientation and toe trajectory|
|32|Right foot index|Right ankle angle; learned foot orientation and toe trajectory|

The classical 36-feature baseline does not use every selected coordinate independently. Elbows and heels contribute to learned trajectories in JEPA. Their inclusion is an explicit anatomical design choice. Face and finger-tip landmarks are excluded. The user did not provide a separate list of feature-to-joint definitions, so this table declares the experimental subset rather than claiming it is uniquely required by S-JEPA or clinically optimal.

Bilateral angle differences summarize image-plane asymmetry. They do not identify the affected side or measure three-dimensional joint mechanics. Keep left/right identities, avoid horizontal reflection augmentation in the main analysis, and do not interpret normalized 2D distances as physical stride length.
