# EMODE: An Emotion Detection Program for Mental Well-being

## Abstract
EmoDe is an application focused on emotion detection using the OpenFace deep learning model for facial action unit analysis. Using inputs such as single persona and multi persona videos and images, the model outputs a video with facial landmarks and analysis of action units (AU), effectively transforming unstructured data to structured data ready for analysis. Action units measure the facial muscle movements defined by the Facial Action Coding System (FACS) in order to quanitfy emotions. For instance, happiness is defined as a nontrivial combination of AUs 6 and 12 (raised cheeks and a pulled corner lip). EmoDe parses through the intensity data for each AU, and determines the emotions displayed based on a given threshold. EMODE's robust capabilities enable a wide range of applications, particularly as human-robot interactions become increasingly prevalent and vital in everyday scenarios.

## Goal

EmoDe will prompt the user with personal questions to elicit an emotional response, and provide a report of what makes the user happy, sad, angry, etc. The questions will focus on a specific topic when an emotion is detected or broaden in scope if no emotional response is observed. The goal of this is to be more aware of what triggers certain emotions in order to make informed decisions about daily activities and interactions to increase emotional well-being. EmoDe will output a report of what topics make the user react with certain emotions, and steps the user can take to improve their mental well-being. 


## Week of 10/8/24
### Goals
1. Download OpenFace and necessary libraries on personal computer to bypass administrative blocks.
2. Pipe the OpenFace output to the VSCode script using webcam as input
------
Found device ID using command line on Powershell: ```Get-PnpDevice | Where-Object { $_.FriendlyName -eq "Integrated Webcam" } | Format-List *```

**Device ID:** 

Computer webcam: USB\VID_0BDA&PID_5556&MI_00\6&2A2E4820&0&0000        
Surface Front: USB\VID_045E&PID_0990&MI_00\6&DB32C28&0&0000       

Used to open webcam feed with OpenFace command line: ```FaceLandmarkVid.exe -device USB\VID_0BDA&PID_5556&MI_00\6&2A2E4820&0&0000```      
```-device <device id> ``` the device ID of a webcam to perform feature extraction from a live feed.     
[OpenFace Webcam Command Line](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments#featureextraction-and-facelandmarkvidmulti)

**How can we store data to run analysis?**

With the command, ```./FeatureExtraction.exe -device "USB\VID_0BDA&PID_5556&MI_00\6&2A2E4820&0&0000"``` , the webcam data was saved to the ```processed``` directory in OpenFace. The data includes an updating csv file with data points, an avi file, and other text documents. We will continue to work with this output to implement it into VSCode. 

In order to implement with VSCode, it will be helpful to be able to control output file names and streamline data for processing time. This following command line was used to in order to specify the output:

```./FeatureExtraction.exe -aus -device 0 -out_dir "C:\Users\cchao2869\Desktop\OpenFace\processed" -of "chao_test"```

where 

```FeatureExtraction``` executable extracts the facial expression features from the video input    
```-aus``` tag extracts only the AU data to save storage and increase processing speed    
```-device 0``` identifies the webcam as the desired video input with index 0 (usually 0 but can be 1, 2, ...)    
```-out_dir``` directs OpenFace on which directory to put the output files     
```-of``` specifies the name of the output files

This allows for control of the file names and what data is extracted compared to the command line using the device ID above. Control of file names, output directory, and data extracted is helpful for the OpenFace implementation in VSCode and saving storage space and increasing processing speed. 


![Screenshot 2024-10-10 135528](https://github.com/user-attachments/assets/937bcef1-dfc0-4733-a2e9-ea705fc61201)
## Week of 9/30/24
### Goals 
1. Finalize application of emotion detection program
2. Pipe the OpenFace output to the VSCode script using webcam as input
------
#### Application Description
The program will prompt the user with personal questions to elicit an emotional response, and provide a report of what makes the user happy, sad, angry, etc. The questions will focus on a specific topic when an emotion is detected or broaden in scope if no emotional response is observed. The goal of this is to be more aware of what triggers certain emotions in order to make informed decisions about daily activities and interactions to increase emotional well-being. It has not yet been determined if the program will simply create a report of what topics make the user react, or will also give advice/solutions.
#### Webcam with OpenFace
While the logic for this process is fairly simple, it proved to be much more difficult than initally expected. In order to use the webcam as input to get real-time emotion detection, the following steps were taken:
1) Run the OpenFace command using the webcam as video input and store the output to test.csv.
2) Open test.csv in AU script using an infinite loop (while True) , so it is always checking the csv file for updates.

   
The UVA Link Lab advisor for the project, Haley Green, provided the following pusedocode for this AU infinite loop:
```python
import time
import csv

def follow_csv(filename):
    # Open the CSV file in read mode
    with open(filename, 'r') as file:
        # Move the pointer to the end of the file to read only new lines
        file.seek(0, 2)
        
        while True:
            # Read the new line
            new_line = file.readline()
            
            if new_line:
                # Convert the line into a CSV row (useful if there are commas, etc.)
                reader = csv.reader([new_line])
                for row in reader:
                    # Print the new row
                    print(row)
            else:
                # Sleep briefly before checking again (to avoid high CPU usage)
                time.sleep(0.5)

# Replace 'your_file.csv' with the path to your actively updating CSV file
follow_csv('/home/haleygreen/OpenFace/build/processed/test.csv')
```

However, this script did not produce any output when run. After troubleshooting both the webcam and OpenFace, it was determined this issue lied in the computer itself. The webcam was effectively opened using a simple code, and OpenFace worked correctly as shown in previous weeks. Futhermore, the following command line was used to streamline the input/output on Powershell: 
```& "C:\Users\cchao2869\Desktop\OpenFace\FaceLandmarkVid.exe" -f 0 -out_dir "C:\Users\cchao2869\Documents\openfaceApp\output"```

This returned an error message that the webcam could not be found at index 0. However, the webcam could not be found at indexes 1, 2, ... either. Due to a variety of administrative locks on the desktop computers, it can be inferred that the administrative locks do not allow the user to continuously create and write a file in a new directory. The simple script that uses the webcam with OpenFace will be run on a personal computer to remove this source of error. 

```python
import time
import csv
import os

def follow_csv(filename):
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        return
    
    # Open the CSV file in read mode
    with open(filename, 'r') as file:
        # Move the pointer to the end of the file to read only new lines
        file.seek(0, 2)
        
        while True:
            try:
                # Read the new line
                new_line = file.readline()
                
                if new_line:
                    # Convert the line into a CSV row (useful if there are commas, etc.)
                    reader = csv.reader([new_line])
                    for row in reader:
                        # Print the new row (or process it)
                        print(row)
                else:
                    # Sleep briefly before checking again (to avoid high CPU usage)
                    time.sleep(0.5)
            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(1)  # Wait a moment before trying again

# Replace 'your_file.csv' with the path to your actively updating CSV file
follow_csv(r'C:\Users\cchao2869\Documents\openfaceApp\test.csv')  # Adjust path as needed
```

Also note that use of the webcam creates several more variables to adjust, including the amount of delay/frames run through the OpenFace model in order to optimize accuracy and speed of results. Once the program can effectively detect emotions using the webcam, the following variables will be considered to optimize performance with the application in mind:

- Use of AU intesity and binary data in unison
- Threshold of AU intensity data
- Delay in loop for webcam lag
- AUs for each emotion
- Amount of time each emotion is displayed
- Delay in emotional response after question is posed


## Week of 9/16/24
### Goals
1. Start using OpenFace on VSCode
2. Research action units (AU) and define emotions based on AU
3. Brainstorm applications of emotion recognition

-----



#### OpenFace on VSCode
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
    'Happiness': (0.5, 0.5),  # AU06_r, AU012_r
    'Sadness': (0.5, 0.5, 0.5),    # AU01_r, AU04_r, AU015_r
    'Anger': (0.5, 0.5, 0.5, 0.5),  # AU04_r, AU07_r, AU05_r, AU23_r
    'Surprise': (0.5, 0.5, 0.5, 0.5),    # AU01_r, AU02_r, AU05_r, AU26_r
    'Fear': (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),   # AU01_r, AU02_r, AU04_r, AU05_r, AU07_r, AU20_r, AU26_r
    'Disgust': (0.5, 0.5, 0.5),     # AU09_r, AU15_r
    'Neutral': (0.2, 0.2)       # All AUs below this threshold
}

# Function to group consecutive frames into time intervals
def get_time_intervals(times, threshold=0.5):
    if not times:
        return []

    times = sorted(float(t) for t in times)
    intervals, start_time = [], times[0]

    for current_time in times[1:]:
        if current_time - start_time > threshold:
            intervals.append((start_time, times[times.index(current_time) - 1]))
            start_time = current_time

    intervals.append((start_time, times[-1]))  # Append the last interval
    return intervals

    
# Function to detect emotions
def detect_emotions_in_seconds(df, fps):
    results = {emotion: [] for emotion in thresholds.keys()}

    for index, row in df.iterrows():
        # Convert frame number to seconds 
        time_in_seconds = row['timestamp']

        # Check for happiness
        if row['AU06_r'] > thresholds['Happiness'][0] and row['AU12_r'] > thresholds['Happiness'][1]:
            results['Happiness'].append(time_in_seconds)

        # Check for sadness
        if row['AU01_r'] > thresholds['Sadness'][0] and row['AU04_r'] > thresholds['Sadness'][1] and row['AU15_r'] > thresholds['Sadness'][2]:
            results['Sadness'].append(time_in_seconds)

        # Check for anger
        if row['AU04_r'] > thresholds['Anger'][0] and row['AU07_r'] > thresholds['Anger'][1] and row['AU05_r'] > thresholds['Anger'][2] and row['AU23_r'] > thresholds['Anger'][3]:
            results['Anger'].append(time_in_seconds)

        # Check for surprise
        if row['AU01_r'] > thresholds['Surprise'][0] and row['AU02_r'] > thresholds['Surprise'][1] and row['AU05_r'] > thresholds['Surprise'][2] and row['AU26_r'] > thresholds['Surprise'][3]:
            results['Surprise'].append(time_in_seconds)

        # Check for fear
        if row['AU01_r'] > thresholds['Fear'][0] and row['AU02_r'] > thresholds['Fear'][1] and row['AU04_r'] > thresholds['Fear'][2] and row['AU05_r'] > thresholds['Fear'][3] and row['AU07_r'] > thresholds['Fear'][4] and row['AU20_r'] > thresholds['Fear'][5] and row['AU26_r'] > thresholds['Fear'][6]:
            results['Fear'].append(time_in_seconds)
        
        # Check for disgust
        if row['AU09_r'] > thresholds['Disgust'][0] and row['AU15_r'] > thresholds['Disgust'][1]:
            results['Disgust'].append(time_in_seconds)

        # Check for neutral (if all AUs are below the neutral threshold)
        if (row['AU01_r'] < thresholds['Neutral'][0] and
            row['AU02_r'] < thresholds['Neutral'][0] and
            row['AU04_r'] < thresholds['Neutral'][0] and
            row['AU05_r'] < thresholds['Neutral'][0] and
            row['AU06_r'] < thresholds['Neutral'][0] and
            row['AU07_r'] < thresholds['Neutral'][0] and
            row['AU09_r'] < thresholds['Neutral'][0] and
            row['AU12_r'] < thresholds['Neutral'][0] and
            row['AU15_r'] < thresholds['Neutral'][0] and
            row['AU20_r'] < thresholds['Neutral'][0] and
            row['AU23_r'] < thresholds['Neutral'][0] and
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
1. Instal OpenFace on Windows
2. Test installation using command lines
3. Start documentation of research on GitHub
-----
#### OpenFace Installation
Installed OpenFace on Windows using [Windows Installation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation). Checked computing using Windows Powershell (32 bit). Also referenced [Tutorial OpenFace Installation](https://youtu.be/e2-Wu_1poBY?si=PI63s-a72uTL974f) to download CEN patch experts (timestamp - 2:30). Used Windows Powershell and command lines available at [OpenFace Command Lines](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments) to test installation.
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

