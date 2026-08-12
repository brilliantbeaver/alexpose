**Role:** You are an expert data scientist specializing in gait analysis and JEPA. 

**Tasks:** You are going to fully understand the "alexpose" repo for how to process the GAVD dataset, you are to map the following gait features to the BLAZEPOSE_33 keypoints that they represent. Process the csv string in the following section: "## Myopathy" to extract keypoints associated with features in column 2 with title "Feature".


Task #1: Extract the data (each feature) from "## Myopathy" 2nd column titled "Feature" into a separate .csv file. In that same file, extract the data from "## Myopathy" 1st column titled "Priority" into a column to the left of the "Feature" column, and input the data correspondingly. 
Create a set of non-duplicate keypoints that are represented by multiple features each. Each BLAZEPOSE_33 region appears at most once as a group header (no duplicates or overlap). Each group should list every feature that traces back to it. 

If a feature's computation spans multiple, unrelated joint regions (whole body, center of mass features, etc.), exclude it from every group instead of listing them in an approximate joint region. 

If two features trace back to the same underlying computation (e.g. one is a literal alias of another, such as stride_time_cv and step_regularity_cv), note that explicitly rather than listing both as if independent.

Task #2: Look at the csv provided below. Using that, look at the second column of the graph, titled "Feature". Extract each of these features and produce a table with columns: Feature|BLAZEPOSE_33 region| Keypoints include


Task #3:
After Task #1 and #2, the output should have some keypoints with more than one feature mapped to it. Create a table of all of the keypoints, in order, as well as the features that are involved with it. 
HOWEVER: Only include the features marked as "H" (high priority) in your table. DO NOT SHOW ANY OTHER KEYPOINTS THAT DO NOT RELATE TO THE HIGH PRIORITY FEATURES.

Your output must complete the following tasks:
The Task #2 table (all features, including excluded ones, one row each).
The Task #1 grouped view: one section per keypoint region, listing its features; followed by an "Excluded (non-region-specific)" section.
The Task #3 de-duped table (all the keypoints and the features relating to them.)
ONLY include the features marked as "H" (high priority) in your table.

## Output Instructions

After creating your analysis, output your results in penny/neuroscience/data as a well structured .md file with the name "PD_keypoint_mapping.md". 

## Myopathy

"Priority 
(H/M/L/NA )",Feature:,Neurological Reason:,What is considered significant?,Source:,AI Check:,
H,left_hip_mean,"Reflects reduced hip extension in terminal stance; hip flexors in myopathy are too weak to eccentrically control (resist) hip extension the way they normally do, since the ankle plantarflexors normally drive hip extension acceleration late in stance",Likely a mean angle of 10 or lower could indicate myopathy,https://pmc.ncbi.nlm.nih.gov/articles/PMC1817673/#TFN1,"High (H). This feature is highly important because hip muscle weakness is a primary characteristic of proximal myopathy, directly causing reduced hip extension during terminal stance as weak hip flexors fail to eccentrically control extension.",same
H,right_hip_mean,"Same mechanism as above; bilateral symmetry expected since myopathy is typically diffuse/systemic, not lateralized",Likely a mean angle of 10 or lower could indicate myopathy,https://pmc.ncbi.nlm.nih.gov/articles/PMC1817673/#TFN1,"High. This feature is considered a high-priority marker because myopathy typically presents as a diffuse, systemic disease causing bilateral muscle weakness, which directly impacts hip extension control during late stance.",same
H,left_knee_mean,Reflects increased knee extension/flexion range as a secondary compensation for proximal (hip) weakness,Likely a mean angle of 50 or higher could indicate myopathy,https://pmc.ncbi.nlm.nih.gov/articles/PMC1817673/#TFN1,"High

The feature left_knee_mean is ranked as high priority because myopathic patients compensate for proximal hip muscle weakness by increasing their knee flexion and extension range of motion to maintain balance and gait efficiency.",same
H,right_knee_mean,"Same mechanism as above; bilateral symmetry expected since myopathy is typically diffuse/systemic, not lateralized",Likely a mean angle of 50 or higher could indicate myopathy,https://pmc.ncbi.nlm.nih.gov/articles/PMC1817673/#TFN1,High. Right knee mean is a high-priority feature because increased knee flexion serves as a key compensatory mechanism to minimize abnormal hip extension caused by proximal muscle weakness.,same
H,walking_speed_ms,"Reduced overall force-generating capacity from proximal muscle weakness directly limits propulsive power, most speed loss commonly attributed to reduced step length rather than cadence in weakness-driven gait",Likely below 0.8 m/s could indicate myopathy as that's considered a slow gait,https://pmc.ncbi.nlm.nih.gov/articles/PMC1817673/#TFN1,"Based on the study, walking speed is ranked as High importance because a reduced force-generating capacity from proximal muscle weakness directly limits propulsive power, leading to a significantly slower gait speed (typically below 0.8 m/s) in myopathy patients.",same
H,step_length_cv,Low CV (near-normal variability) despite reduced magnitude reflects a mechanically consistent strength deficit rather than an unstable motor-planning problem,"A cv equivalent with a normal person's cv (so low), in combination with these other features would most likely indicate myopathy","My theory primarily, based on my understanding of what myopathy is
https://www.ncbi.nlm.nih.gov/books/NBK562290/","High (H). A low step length coefficient of variation reflects a consistent, mechanically driven strength deficit from bilateral muscle weakness rather than an unstable, variable motor-planning or neurological coordination issue.",same
M,stride_length_m,"This is a symptom of myopathy in some but not all patients, as a result of their reduced muscle strength","Likely below 0.9 m could indicate myopathy, but the lack of an abnormal stride length should not rule out myopathy either",https://pmc.ncbi.nlm.nih.gov/articles/PMC1817673/#TFN1,"Medium. Stride length typically decreases as muscular weakness progresses in myopathic patients, but it is considered a secondary symptom that may not be abnormal in all cases.",same
H,trunk_lean_angle,"When muscles of the pelvic girdle and spine are weakened, as a result the patient will lean their trunk abnormally forward to compensate for balance.",An angle above 10 degrees could likely indicate myopathy,https://pmc.ncbi.nlm.nih.gov/articles/PMC2949322/,"The study indicates that compensatory forward trunk lean (such as an angle above 10 degrees) occurs to maintain balance and reduce knee joint load when proximal muscle weakness is present, making this feature a high (H) priority for diagnosing myopathic gait.",same
H,positional_symmetry_score,"Similar logic to a lower CV, if the positional symetry score is lower (what a normal person would get), this implies that there is full body muscle weakness, which is what myopathy is, rather than unilateral weakness that is present with stroke, cerebral palsy, and PD in its earlier stages",A lower score (not clarified in feature tutorial doc),"My theory primarily, based on my understanding of what myopathy is
https://www.ncbi.nlm.nih.gov/books/NBK562290/","High (H). Since myopathies typically present with symmetric, bilateral muscle weakness rather than unilateral weakness, a normal positional symmetry score helps distinguish it from unilateral conditions.",same
H,postural_sway_area,"Due to myopathy causing muscle weakness, patients would have a harder time finding balance, leading to an overall postural sway.",A higher area (specific numbers not clarified in feature tutorial doc),https://www.ncbi.nlm.nih.gov/books/NBK562290/,"High. Proximal muscle weakness, a hallmark symptom of myopathy, directly impairs pelvic and lower limb stability, making postural sway a key indicator of compromised balance.",same
