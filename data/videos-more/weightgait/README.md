Welcome to the WeightGait Dataset. The following is a description of the file structure of the data. For implementing this in Python, please see getting_started.py. 
For the codebase this dataset was developed with, see https://github.com/ChrisLochhead/GaitMonitor.


#Dataset Structure

The data is stored as sets of images, both RGB with the corresponding depth images stored in the same folder. For example, Instance_n.0 will have M frames of RGB data, then M frames of the corresponding D data.

There is also a joint data folder which contains the extracted joints in the same order, extracted by HigherHRNet, following the COCO joint format. See the whole body examples in https://github.com/CMU-Perceptual-Computing-Lab/openpose
for an example of the joints. 
17 joints are extracted (***).
All joints are in raw form and need pre-processing to convert to (x,y,z) format using
the function rs2_deproject_pixel_to_point as discussed below.

Each row corresponds to one frame, with the following structure. 
Column  Content            Coding
1       sequence instance  0..1919
2       frame number       A..B
3       weight class       0:no weights 1:left leg 2:both legs
4       has gait freeze    0:no 1:yes
5       has obstacle       0:no 1:yes
6       person ID          0..31
7-23    17 joint positions "[image-col image-row depth]"

##File Structure

Datasets/
	All Depth Videos/
		Instance_1.0/
		.
		.	
		.
		Instance_1919.0/	

	All Joints/
		Absolute_Data.csv

	All RGB Videos/
		Instance_1.0/
		.
		.	
		.
		Instance_1919.0/
	
	Example Videos/
		Instance_1.0/
		.
		.	
		.
		Instance_1919.0/	

#Getting Started

To get started you can view the repo at https://github.com/ChrisLochhead/GaitMonitor and look at process.py in the main directory. This contains functions for loading in the data and running machine learning models on it. Place
"Datasets" in the same folder as process.py, however do note that you will need to change the file names and paths as the directory as of writing is an active working zone where different versions of the joint_data are being used which
are named different things other than data.csv.

All Joint values are in the format [column, row, depth] in pixel values. To convert these to joint co-ordinates, we recommend using the Pyrealsense2 package 2, and using the "rs2_deproject_pixel_to_point" function. The
intrinsics of the Intel Realsense D415 Camera used are as follows: 

212	
120	
107.02976989746094	
61.70092010498047	
154.89523315429688	
154.63319396972656	
distortion.inverse_brown_conrady	
[0.0, 0.0, 0.0, 0.0, 0.0]

The file for this can be found in the GaitMonitor Repo at: https://github.com/ChrisLochhead/GaitMonitor/blob/master/Code/depth_intrinsics.csv The average frame rate during recording varied, but was typically between 27 - 32 FPS.


The recorded joints follow the Coco format with 17 joints preceded by the 6 metadata values in the following order:

"['Instance', 'No_In_Sequence', 'Class', 'Freeze', 'Obstacle', 'Person', 'Nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
    'L_elbow','R_elbow','L_hand','R_hand','L_hip','R_hip','L_knee','R_knee','L_foot', 'R_foot'] 


If you have any queries about using the dataset or the platform, contact s2057496@ed.ac.uk and I'd be glad to help.
 
