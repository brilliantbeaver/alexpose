# The exact joint contract

The user brief names features and asks for deduplication, but does not supply a literal list of keypoints. This project therefore declares a transparent feature-derived set. We start with the hip, knee, and ankle geometry used in exp5, then include arm swing, elbow flexion, and foot-axis orientation. Their union has **16 unique BlazePose landmarks**. This is a study choice, not an anatomical set mandated by the S-JEPA paper.

|BLAZEPOSE_33 index|Keypoint name|Features involved|
|---:|---|---|
|11|Left shoulder|Left hip angle; left elbow angle; context torso scale|
|12|Right shoulder|Right hip angle; right elbow angle; context torso scale|
|13|Left elbow|Left elbow angle|
|14|Right elbow|Right elbow angle|
|15|Left wrist|Left elbow angle; left arm swing speed|
|16|Right wrist|Right elbow angle; right arm swing speed|
|23|Left hip|Left hip and knee angles; pelvis center; context torso scale|
|24|Right hip|Right hip and knee angles; pelvis center; context torso scale|
|25|Left knee|Left hip, knee, and ankle angles|
|26|Right knee|Right hip, knee, and ankle angles|
|27|Left ankle|Left knee and ankle angles; ankle separation|
|28|Right ankle|Right knee and ankle angles; ankle separation|
|29|Left heel|Left foot-axis inclination|
|30|Right heel|Right foot-axis inclination|
|31|Left foot index|Left ankle angle; left foot-axis inclination|
|32|Right foot index|Right ankle angle; right foot-axis inclination|

`worldmodels.data.JOINT_IDS` is the numerical source of truth. After selection, tensor position 0 means BlazePose joint 11, and tensor position 15 means BlazePose joint 32. Tensor position 11 is **not** BlazePose joint 11. Model configuration and every mask must preserve this mapping. Neither the face nor finger landmarks enter the predictor or its masking universe.

A token is one joint in one frame. A 16-frame window has 256 candidate positions. Missing observations retain a separate Boolean validity mask. Uniform masking draws from eligible observed positions without consulting velocity, motion magnitude, condition labels, or confidence magnitude. Confidence only determines whether an observation is valid. It never makes an observed fast joint more likely to be masked.

## What the features measure

For a hip angle, the three points are shoulder, hip, and knee. For a knee angle, they are hip, knee, and ankle. For an ankle angle, they are knee, ankle, and foot index. The middle point is the vertex. The shared `ambient.pose.joint_angles.get_joint_angles` implementation supplies these six projected angles.

The first 15 Random Forest features preserve exp5's core design: six mean angles, three left-right differences in those means, and six ranges. The extension adds six angle standard deviations, six elbow summaries, four projected foot-axis summaries, four wrist-speed summaries, and two ankle-separation summaries. This yields 37 measured fields. We do not describe this as exp5's 82-feature configuration, and we do not compare scores from unmatched cohorts as if they were a controlled experiment.

Positions are measured in image pixels before normalization. Each frame is centered on its pelvis midpoint. The median observed torso length in the first eight context frames sets one scale for the whole window. Future frames cannot change the visible context through scale estimation. A missing pelvis invalidates its frame. Missing coordinates become zeros only after a separate validity array records their missingness.

The resulting units are torso lengths. A wrist change divided by the measured sampling interval is a projected relative-motion rate in torso lengths per second. It is not walking speed in metres per second. These features describe appearance in the image plane, not measured 3D anatomy. Centering removes global body translation, so this experiment forecasts relative articulation rather than where the person will travel through the room.

## Missing data and quality

The tutorial confidence cutoff is 0.5 and its default minimum observed fraction is 0.7. Both are declared study settings, not Ambient defaults. No interpolation turns missing landmarks into observed targets. The RF uses the identical windows and observed joints as JEPA. A feature with no valid observations is NaN and is imputed using training data only. The legacy `GaitFeatureVector.from_joint_angles` convenience method substitutes zero for some missing angles, so this experiment obtains the shared angle series and preserves missingness explicitly.

## Preprocessing and masked completion

Normalization is computed before a random training mask is drawn. Therefore a masked hip or shoulder may already have contributed to the common pelvis/torso transformation. Random masking here trains completion of a pre-normalized articulation sequence; it is not a claim that the original hidden pixel coordinates have no influence anywhere in preprocessing. A strict arbitrary-mask conditional task would need a transform estimated only from the actual visible landmarks, with its own missing-anchor policy and run identity. Prefix-only scale estimation does prevent future frames from changing the visible prefix for forecasting, but it does not remove the preceding random-mask distinction.
