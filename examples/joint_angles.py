from ambient.gavd import GAVDDataLoader
from ambient.pose.joint_angles import get_joint_angles as calculate_angles

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == "__main__":

    project_root = Path.cwd()
    print(f"==> project_root: {project_root}")

    # 1. Load GAVD gait sequences
    loader = GAVDDataLoader()
    ONE_SEQUENCE_PATH = Path(project_root, "data", "GAVD_Clinical_Annotations_1.1.csv")
    df = loader.load_gavd_data(ONE_SEQUENCE_PATH)
    sequences = loader.organize_by_sequence(df)
    sequence_id = list(sequences.keys())[1]
    sequence_data = sequences[sequence_id]
    print(f"\tnum_sequences: {len(sequences)}")

    # 2. Extract body keypoints (pose landmarks)
    keypoints_array = get_keypoints(project_root=project_root, sequence_data=sequence_data)

    # 3. Compute joint angles – For each frame, calculate hip, knee and ankle angles

    joint_angles_array = calculate_angles(
        keypoints_array=keypoints_array,
        keypoint_format="BLAZEPOSE_33",
        fps=30.0,
        confidence_threshold=0.3
    )

    print(f"==> # keypoints: {len(keypoints_array)}")
    print(f"    # joint angles: {len(joint_angles_array)}")
    print(f"The first set of these joint angles look like: {joint_angles_array[0]}")
