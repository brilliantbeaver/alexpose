from pathlib import Path

from ambient.gavd import GAVDDataLoader
from ambient.pose.joint_angles import get_joint_angles as calculate_angles
from ambient.pose.keypoints import SequenceKeypointExtractor
from ambient.utils.eval_keypoints import get_gavd_frame

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
    print(f"\tusing sequence: {sequence_id}")

    # 2. Extract body keypoints (pose landmarks)
    # get_keypoints now only accepts DataFrame, returns (keypoints, first_frame) tuple
    extractor = SequenceKeypointExtractor()
    keypoints_array = extractor.extract_from_sequence(
            sequence_data=sequence_data,
            video_base_path=project_root / "data" / "youtube",
    )
 
    # 3. Compute joint angles – For each frame, calculate hip, knee and ankle angles
    joint_angles = calculate_angles(
        keypoints_array=keypoints_array,
        keypoint_format="BLAZEPOSE_33",
        fps=30.0,
        confidence_threshold=0.3
    )

    print(f"==> # keypoints: {len(keypoints_array)} frames")
    print(f"    # joint angles: {len(joint_angles.frames)} frames")
    print(f"\nFirst frame joint angles:")
    for joint_name, angle_data in joint_angles.frames[0].angles.items():
        print(f"  {joint_name}: {angle_data.angle_degrees:.2f}° (confidence: {angle_data.confidence:.2f})")
