# Abstract
EMODE is an application focused on emotion detection using the OpenFace deep learning model for facial action unit analysis. Using inputs such as single persona and multi persona videos and images, the model outputs a video with facial landmarks and analysis of action units (AU), effectively transforming unstructured data to structured data ready for analysis. Action units measure the facial muscle movements defined by the Facial Action Coding System (FACS) in order to quanitfy emotions. For instance, happiness is defined as a nontrivial combination of AUs 6 and 12 (raised cheeks and a pulled corner lip). EMODE parses through the intensity data for each AU, and determines the emotions displayed based on a given threshold. EMODE's robust capabilities enable a wide range of applications, particularly as human-robot interactions become increasingly prevalent and vital in everyday scenarios.

## Week of 9/16/24
### Goals
- Research action units (AU) and define emotions based on AU
- Brainstorm applications of emotion recognition
- Start using OpenFace on VSCode
-----

In order to effectively manipulate the OpenFace data, couple of libraries were installed. These included ```cv2``` and ```pandas```.  ```cv2``` is part of the OpenCV library, which o used for computer vision tasks. It provides tools for image and video processing, including functions for image manipulation, feature detection, and object recognition. The ```pandas``` library, imported as ```pd```, is s powerful data manipulation and analysis library that provides data structures like DataFrames. DataFrames are two-dimensional, size-mutable, and potentially heterogeneous tabular data structure that are very helpful for machine learning tasks due to their versatility and user-friendly nature. The ```pandas``` library is widely used for data cleaning, transformation, and analysis, making it easier to work with structured data.



``` python
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



# Read the CSV file into a DataFrame (table format)
df = pd.read_csv(csv_file)

# Clean column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()

# Set your thresholds for each AU
thresholds = {
    'Happiness': (0.5, 0.5),  # AU01_r, AU06_r
    'Sadness': (0.5, 0.5),    # AU01_r, AU04_r, AU015_r
    'Anger': (0.5, 0.5, 0.5),  # AU04_r, AU07_r, AU05_r
    'Surprise': (0.5, 0.5),    # AU05_r, AU26_r
    'Fear': (0.5, 0.5, 0.5),   # AU01_r, AU02_r, AU05_r
    'Neutral': (0.2, 0.2)       # All AUs below this threshold
}

# Function to group consecutive frames into time intervals
def get_time_intervals(times, threshold=1/fps):
    """Group consecutive times into intervals based on a threshold for time difference."""
    if not times:
        return []
    
    times = sorted([float(t) for t in times])  # Convert times to float and sort them
    intervals = []
    start_time = times[0]

    for i in range(1, len(times)):
        if times[i] - times[i - 1] > threshold:  # Check if gap between times exceeds threshold
            intervals.append((start_time, times[i - 1]))  # Create a new interval
            start_time = times[i]
    
    intervals.append((start_time, times[-1]))  # Append the last interval
    return intervals

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

    # Convert the results into intervals of consecutive frames in seconds
    interval_results = {emotion: get_time_intervals(times) for emotion, times in results.items()}
    
    return interval_results

# Detect emotions and group them into time intervals
emotion_intervals = detect_emotions_in_seconds(df, fps)

# Print the results
for emotion, intervals in emotion_intervals.items():
    if intervals:
        print(f"{emotion} detected at intervals: {intervals}")
    else:
        print(f"No {emotion} detected.")
```

Coding on VSCode with OpenFace was easier than expected, as the process was similar to using command lines on PowerShell. As we look forward and begin to brainstorm ideas for the application of this emotion detection program, it is important to note that data collection and analysis is key to determine the accuracy of this model. We have created a preliminary map of action units to emotions available in the README at: [Action Units to Emotions](https://github.com/Jpark27614/Emotion-Detection-Program/blob/main/README.md#action-unit-combinations-and-emotions).  To change the results of the model, the combinations of action units can be changed. In addition, the thresholds for every active unit to be "active" can be easily changed. With additional research, we can determine which AUs are more important than others in order to get an accurate detection. 


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
