
## Week of 9/9/24
### Goals
- Instal OpenFace on Windows
- Test installation using command lines
- Start documentation of research on GitHub
-----
#### OpenFace Installation
Installed OpenFace on Windows using [Windows Installation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation). Checked computing using Windows Powershell (32 bit). Used Windows Powershell and command lines available at [OpenFace Command Lines](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments) to test installation.
Note that videos must be in mp4 format to run through OpenFace application. The following command lines were used to run sequence analysis (locate facial points) on a video with one person (command line argument: ```FaceLandmarkVid```): 

``` ruby
# Locate OpenFace Directory
cd "C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86"

# Places landmarks on the video for Facial Recognition (FR) 
.\FaceLandmarkVid.exe -f "C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86\example_video.mp4" -out_dir "C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\output"

```

This resulted in an avi file with the designed FR points, and a large data table (csv file). 

https://github.com/user-attachments/assets/06a0686e-6d9e-4fc9-a614-468a45c967a4

An additional test was performed using multiple faces. The following command line was implented: 

```ruby
# Places landmarks on the video for multiple faces
.\FaceLandmarkVidMulti.exe -f "C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86\example_video3.mp4" -out_dir "C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\output"
```

https://github.com/user-attachments/assets/33e166a9-8f0f-49f1-82b5-e32fe2e96587


## Week of 9/16/24
### Goals
- Research action units (AU) and define emotions based on AU
- Brainstorm applications of emotion recognition
- Start using OpenFace on VSCode
-----

(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7264164/)


``` ruby
import subprocess
import os
import csv
import pandas as pd
import cv2




# Set up paths
openface_dir = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86'  # Update this path
video_file = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86\example_video4.mp4'
output_dir = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\outputVS'


# Open the video file
video = cv2.VideoCapture(video_file)

# Get the FPS (Frames Per Second)
fps = video.get(cv2.CAP_PROP_FPS)

# Print the FPS value
print(f"Frames per second (FPS): {fps}")

# Release the video file
video.release()


# Define the command
command = [
    os.path.join(openface_dir, 'FeatureExtraction.exe'),  # OpenFace binary
    '-f', video_file,  # Input video
    '-out_dir', output_dir  # Output directory
]

# Run the command
subprocess.run(command)

# Path to your CSV file
csv_file = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\outputVS\example_video4.csv'



# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Clean column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()



# Set your thresholds for each AU
thresholds = {
    'Happiness': (0.5, 0.5),  # AU01_r, AU06_r
    'Sadness': (0.5, 0.5),    # AU01_r, AU04_r
    'Anger': (0.5, 0.5, 0.5),  # AU04_r, AU07_r, AU05_r
    'Surprise': (0.5, 0.5),    # AU05_r, AU26_r
    'Fear': (0.5, 0.5, 0.5),   # AU01_r, AU02_r, AU05_r
    'Neutral': (0.2, 0.2)       # All AUs below this threshold
}

# Function to detect emotions
def detect_emotions_in_seconds(df, fps):
    results = {emotion: [] for emotion in thresholds.keys()}

    for index, row in df.iterrows():
        # Convert frame number to seconds 
        time_in_seconds = row['timestamp']

        # Check for happiness
        if row['AU01_r'] > thresholds['Happiness'][0] and row['AU06_r'] > thresholds['Happiness'][1]:
            results['Happiness'].append(time_in_seconds)

        # Check for sadness
        if row['AU01_r'] > thresholds['Sadness'][0] and row['AU04_r'] > thresholds['Sadness'][1]:
            results['Sadness'].append(time_in_seconds)

        # Check for anger
        if row['AU04_r'] > thresholds['Anger'][0] and row['AU07_r'] > thresholds['Anger'][1] and row['AU05_r'] > thresholds['Anger'][2]:
            results['Anger'].append(time_in_seconds)

        # Check for surprise
        if row['AU05_r'] > thresholds['Surprise'][0] and row['AU26_r'] > thresholds['Surprise'][1]:
            results['Surprise'].append(time_in_seconds)

        # Check for fear
        if row['AU01_r'] > thresholds['Fear'][0] and row['AU02_r'] > thresholds['Fear'][1] and row['AU05_r'] > thresholds['Fear'][2]:
            results['Fear'].append(time_in_seconds)

        # Check for neutral (if all AUs are below the neutral threshold)
        if (row['AU01_r'] < thresholds['Neutral'][0] and
            row['AU02_r'] < thresholds['Neutral'][0] and
            row['AU04_r'] < thresholds['Neutral'][0] and
            row['AU05_r'] < thresholds['Neutral'][0] and
            row['AU06_r'] < thresholds['Neutral'][0] and
            row['AU07_r'] < thresholds['Neutral'][0] and
            row['AU26_r'] < thresholds['Neutral'][0]):
            results['Neutral'].append(time_in_seconds)

    return results

# Detect emotions in the video
emotion_results_in_seconds = detect_emotions_in_seconds(df, fps)

# Print the results
for emotion, times in emotion_results_in_seconds.items():
    if times:
        print(f"{emotion} detected at seconds: {times}")
    else:
        print(f"No {emotion} detected.")
```

Coding on VSCode with OpenFace was easier than expected, as the process was similar to using command lines on PowerShell. As we look forward and begin to brainstorm ideas for the application of this emotion detection program, it is important to note that data collection and analysis is key to determine the accuracy of this model. We have created a preliminary map of action units to emotions available in the README at: [Action Units to Emotions](https://github.com/Jpark27614/Emotion-Detection-Program/blob/main/README.md#action-unit-combinations-and-emotions).  To change the results of the model, the combinations of action units can be changed. In addition, the thresholds for every active unit to be "active" can be easily changed. With additional research, we can determine which AUs are more important than others in order to get an accurate detection. 
