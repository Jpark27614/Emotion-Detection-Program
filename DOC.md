
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

This resulted in an avi file with the designed FR points, and a large data table. 

https://github.com/user-attachments/assets/22897eb1-97b1-4257-bc90-ea6a54b04863

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


# Set up paths
openface_dir = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86'  # Update this path
video_file = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86\example_video2.mp4'
output_dir = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\outputVS'

# Define the command
command = [
    os.path.join(openface_dir, 'FaceLandmarkVidMulti.exe'),  # OpenFace binary
    '-f', video_file,  # Input video
    '-out_dir', output_dir  # Output directory
]

# Run the command
subprocess.run(command)

# Path to your CSV file
csv_file = r'C:\Users\cchao2869\Desktop\OpenFace_2.2.0_win_x86\outputVS\example_video2.csv'

# Open and read the CSV file
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    
    # Skip the header if necessary
    header = next(reader)

    # Iterate through the rows
    for row in reader:
        print(row)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# List all columns (helps you find where the AU data starts)
print(df.columns)

# Extract intensity columns (r for regressor)
au_intensity_columns = [col for col in df.columns if '_r' in col]
print("Action Unit Intensity Columns:", au_intensity_columns)

# Example: Extract AU12 (lip corner puller) intensities
au12_intensity = df['AU12_r']
print("AU12 Intensity Values:", au12_intensity)
```

