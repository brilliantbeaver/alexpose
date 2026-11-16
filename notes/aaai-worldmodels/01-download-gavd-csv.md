# Role

You are an expert CSV data processor for managing gait analysis data for AI/ML applications

# Tasks

You are to first systematically understand the GAVD dataset (CSV files) already downloaded into "data/gavd", particularly read the dataset summary at: "data/gavd/README.md".  The GAVD gait sequences have already been grouped by "gait_pat" into distinct sub-directories of "data/gavd".

For each gait sequence, if not already, download the respective Youtube video from the following "gait_pat" categories: "cerebral palsy", "myopathic", "normal", "parkinsons", and "stroke", to the corresponding subfolder of "data/videos-gavd" grouped by the "gait_pat".  

Then, extract the associated frames for a given sequence and save the cropped frames of video into the corresponding "gait_pat" subfolder of: "data/videos-ms/gavd-sequences".  These cropped video files must only include the segment of the youtube video indicatd in the GAVD csv file starting from the first frame listed to the ending frame only, inclusive.  

The stored GAVD gait sequence videos should be organized into folders according to the "gait_pat" annotated gait conditions included in each ".csv" file. Name each video file following this format:"<youtube id>-<starting frame>-<ending frame>". 

For example, from this GAVD csv file: "data/gavd/normal/cljo2wwu7001a3n6ljmqm39l6.csv", the saved output sequence should be: "data/videos-ms/gavd-sequences/normal/3KnFt8bH3tE-16-111.mp4"

After you are done, thoroughly document what has been downloaded and processed in a well-structured markdown file: "data/videos-ms/gavd-sequences/README.md" file with the completed downloads, extractions, frames, sequences, and organization.

Use adversarial review to check and verify your work.  Fix all issues systematically and thoroughly.

Fan out subagents with dynamic workflows.

---

When writing the final README.md, use plain natural languages, complete sentences, and avoid using staccato cadence, and avoid using syllogism or jargons.  If illustration with vector graphics is helpful, generate modern vivid colored vector graphics illustration to help make the extractin process clear and easy to understand.

--- 
 
# ROLE
You are an expert orchestrator of a team of research agents. 

# TASK
Ultrathink: You are to systematically and thoughtfully create a team of research agents to carry out the tasks outlined in: "02-create-initial-tutorials.md". Deeply research on the best agentic coding practices and apply them to the creation of this team of research agents. 

You should persist these agent definitions in the local project. 