**Role:** You are an expert data scientist specializing in gait analysis and JEPA. 

**Tasks:** You are going to fully understand the "alexpose" repo for how to process the GAVD dataset, you are to map the following gait features to the BLAZEPOSE_33 keypoints that they represent. Process the csv string in the following section: "## Cerebral Palsy" to extract keypoints associated with features in column 2 with title "Feature".


Task #1: Extract the data (each feature) from "## Cerebral Palsy" 2nd column titled "Feature" into a separate .csv file. In that same file, extract the data from "## Cerebral Palsy" 1st column titled "Priority" into a column to the left of the "Feature" column, and input the data correspondingly. 
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

## Cerebral Palsy

"Priority 
(H/M/L/NA )",Feature:,Neurological Reason:,What is considered significant?,Source:,AI Check:,
H,left_knee_mean,"Spasticity of the plantarflexors combined with weakness/lengthening of the calf muscles disrupts the normal ""plantarflexion-knee extension couple"" that keeps the knee extended in stance, driving compensatory crouch","Classically defined as knee flexion of at least 30 degrees throughout the stance phase (Sutherland & Davids threshold); a 2023 study found children at GMFCS level II averaged ~20° knee flexion in stance, while GMFCS III averaged ~40°",https://pmc.ncbi.nlm.nih.gov/articles/PMC10741394/,"High
Knee flexion during walking is highly important because a flexed knee gait (crouch gait) is a common and clinically significant deformity in children with cerebral palsy that heavily impacts their long-term walking ability and functional mobility.",same
H,right_knee_mean,Same concept as left knee,Same 30° stance-phase threshold applies to each limb independently,https://pmc.ncbi.nlm.nih.gov/articles/PMC10741394/,"The feature right_knee_mean is ranked as high priority (H) because quantifying stance-phase knee flexion is critical for identifying crouch gait, a major clinical sign of spastic cerebral palsy.",same
H,left_ankle_mean,Spastic/contractured plantarflexors (true equinus) vs. secondary compensatory dorsiflexion (crouch) reflect two distinct disrupted motor control pathways in spastic CP,Excessive dorsiflexion combined with excessive knee/hip flexion defines crouch gait; ankle in plantarflexion throughout stance with extended hip/knee defines true equinus,https://pmc.ncbi.nlm.nih.gov/articles/PMC5489760/,High. Left ankle kinematics are of high importance because distinguishing between true equinus and compensatory crouch helps isolate distinct disrupted motor control pathways essential for planning effective clinical and surgical interventions.,same
H,right_ankle_mean,Same concept as left ankle,"Same criteria as above, applied per limb",https://pmc.ncbi.nlm.nih.gov/articles/PMC5489760/,High (H) because ankle kinematics are critical for distinguishing key primary gait deviations like true equinus and crouch gait in spastic cerebral palsy.,same
H,left_hip_mean,"Impaired selective motor control and spasticity of hip flexors (iliopsoas) prevents full hip extension in stance, part of the same proximal-to-distal disruption chain as crouch/jump gait","A 2025 prevalence study found reduced hip extension was significantly associated with crouch gait (p=0.002), alongside greater knee flexion (p<0.001)",https://jhwcr.com/index.php/jhwcr/article/view/753/678,"High

Reduced hip extension is ranked as highly important because it is a key component of the proximal-to-distal kinematic disruption chain in spastic cerebral palsy and is significantly associated with crouch gait.",same
H,right_hip_mean,Same concept as left hip,Same p=0.002 association applies per limb,https://jhwcr.com/index.php/jhwcr/article/view/753/678,"High. The right hip mean is ranked as a high priority feature because a 2025 prevalence study demonstrated that reduced hip extension is significantly associated with crouch gait (p=0.002), which is a major pathological walking pattern in children with cerebral palsy.",same
H,knee_asymmetry,"Reflects whether the underlying brain injury is unilateral (hemiplegic CP, lateralized corticospinal damage) vs. bilateral (diplegic/quadriplegic CP, more symmetric involvement)","Presence of significant inter-limb asymmetry supports a hemiplegic classification; researchers added a distinct ""asymmetric gait group"" when the two limbs fell into different classifications entirely",https://pmc.ncbi.nlm.nih.gov/articles/PMC5489760/,"High

Knee asymmetry is highly important because the presence of significant inter-limb asymmetry supports a hemiplegic classification, helping clinicians distinguish between unilateral (hemiplegic) and bilateral (diplegic/quadriplegic) cerebral palsy.",same
H,pelvic_tilt_mean,Compensatory anterior pelvic tilt develops to keep the center of mass balanced when hip extensors are weak/spastic and knees are chronically flexed,"Anterior pelvic tilt and increased lumbar lordosis are defining features of jump gait; one surgical outcomes study found mean anterior pelvic tilt increased by 9.9° post-intervention as knee flexion improved, confirming the mechanical link",https://pmc.ncbi.nlm.nih.gov/articles/PMC6780050/,High. Anterior pelvic tilt is a defining feature of jump gait and is mechanically linked to knee flexion improvements as seen in surgical outcomes.,same
H,walking_speed_ms,"Reduced motor efficiency from spasticity/crouch increases the metabolic and mechanical cost of walking, forcing a slower, more cautious gait","A case report found walking speed reached its slowest point concurrently with worsening crouch, increased postural sway, and greatest walking energy expenditure",https://pmc.ncbi.nlm.nih.gov/articles/PMC4908800/,"High

Based on the study, walking speed is a highly critical gait parameter because its decline directly correlates with worsening crouch, increased postural instability, and a significantly higher energy expenditure in children with cerebral palsy.",same
H,stride_length_m,Shortened stance-phase extension (from crouch) mechanically limits how far the leg can propel the body forward each step,A 2025 study found significantly shorter steps and strides in children with crouch gait (both p<0.001),https://jhwcr.com/index.php/jhwcr/article/view/753/678,High. A 2025 study found that children with crouch gait have significantly shorter step and stride lengths (p<0.001) due to shortened stance-phase extension mechanically limiting forward propulsion.,same
