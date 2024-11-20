# EmoDe: An Emotion Detection Program for Mental Well-being

## Abstract
EmoDe is an application focused on emotion detection using the OpenFace deep learning model for facial action unit analysis. Using inputs such as single persona and multi persona videos and images, the model outputs a video with facial landmarks and analysis of action units (AU), effectively transforming unstructured data to structured data ready for analysis. Action units measure the facial muscle movements defined by the Facial Action Coding System (FACS) in order to quanitfy emotions. For instance, happiness is defined as a nontrivial combination of AUs 6 and 12 (raised cheeks and a pulled corner lip). EmoDe parses through the intensity data for each AU, and determines the emotions displayed based on a given threshold. EMODE's robust capabilities enable a wide range of applications, particularly as human-robot interactions become increasingly prevalent and vital in everyday scenarios.


EmoDe will prompt the user with personal questions to elicit an emotional response, and provide a report of what makes the user happy, sad, angry, etc. The questions will focus on a specific topic when an emotion is detected or broaden in scope if no emotional response is observed. The goal of this is to be more aware of what triggers certain emotions in order to make informed decisions about daily activities and interactions to increase emotional well-being. EmoDe will output a report of what topics make the user react with certain emotions, and steps the user can take to improve their mental well-being. 


https://github.com/user-attachments/assets/0549635f-c8b5-40c8-9dfa-6d20df7d3eaa

### SER Model
While we are working on rule-based classification, after researching the pyAudioAnalysis library, we have found sufficient documentation for how to create a machine learning algorithm. Implementing rule-based classification would require extracting features and running an experiment to manually standardize audio features which might be more time consuming. In addition, there was significantly more documentation on pyAudioAnalysis for training and testing models than feature extraction. Once we decided to train and test a machine learning algorithm, we had to decide which one to implement. The two easiest algorithms to use with the pyAudioAnalysis library are kNNs and SVMs. These are described below:       

| **Aspect**       | **k-Nearest Neighbors (kNN)**                                                                                                                                                    | **Support Vector Machines (SVM)**                                                                                                                                 |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Short Definition** | A simple algorithm that classifies data points based on the **majority class of the k nearest neighbors** using a chosen distance metric.                                       | A powerful classifier that finds the **optimal hyperplane** to separate different classes, using a **kernel trick** if necessary to transform data into higher dimensions. |
| **Example** | Imagine a new fruit whose **weight** and **color** need to be classified as an apple or an orange. kNN would look at the **k nearest existing fruits** and classify it based on the majority. | The SVM would find a **line (or hyperplane)** that best separates apples from oranges in terms of weight and color, ensuring the maximum margin between classes.          |
| **Pros**           | - Easy to implement and understand  <br> - No training phase required  <br> - Flexible to various distance metrics                                                               | - Effective in high-dimensional spaces <br> - Works well with clear class separation <br> - Can use different kernel functions for non-linear classification            |
| **Cons**           | - Slow for large datasets, as it computes distances for every data point <br> - Sensitive to the choice of k and scaling of features <br> - Requires storing the entire dataset | - Can be complex to implement and train <br> - Sensitive to parameter settings <br> - Less effective with noisy or overlapping data, and computationally intensive   |


We decided to train an SVM due to the multiclass classification of emotion detection (classes = happy, sad, angry, surprised, fearful, disgust, neutral). We used the [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). The code to train and test the model is shown below (note that data must be organized in directories for each class in order to train and test models with pyAudioAnalysis). 

```python
from pyAudioAnalysis.audioTrainTest import extract_features_and_train
mt, st = 1.0, 0.05
dirs = [
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\neutral",
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\happy",
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\sad",
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\angry",
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\fearful",
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\disgust",
    r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\data\surprised"
]

extract_features_and_train(dirs, mt, mt, st, st, "svm", "svm_emotions")
```

After the training was complete, the following was printed: 


```
                neutral                 happy                   sad                     angry                   fearful                 disgust                 surprised       OVERALL
        C       PRE     REC     f1      PRE     REC     f1      PRE     REC     f1      PRE     REC     f1      PRE     REC     f1      PRE     REC     f1      PRE     REC     f1       ACC     f1
        0.001   0.0     0.0     0.0     35.7    27.9    31.3    34.3    54.8    42.2    49.4    54.8    52.0    44.4    38.0    40.9    36.6    49.1    41.9    50.7    42.2    46.1     41.0    36.3
        0.010   33.1    43.0    37.4    40.6    34.5    37.3    42.6    47.0    44.7    60.0    60.4    60.2    48.0    51.4    49.7    46.9    48.4    47.7    57.9    46.0    51.3     47.6    46.9
        0.500   35.1    50.4    41.4    34.7    42.4    38.2    39.3    40.6    39.9    53.8    53.9    53.9    46.9    44.8    45.9    56.0    41.2    47.5    56.2    46.9    51.2     45.3    45.4
        1.000   34.5    48.5    40.3    42.1    49.1    45.3    43.5    44.8    44.1    53.1    57.1    55.0    54.5    47.2    50.6    50.9    42.5    46.3    56.0    45.6    50.3     47.7    47.4     best f1         best Acc
        5.000   30.3    48.7    37.4    40.2    48.2    43.9    38.5    40.3    39.4    49.7    49.8    49.8    47.9    41.1    44.3    52.4    41.5    46.3    54.5    43.9    48.6     44.5    44.2
        10.000  32.6    51.4    39.9    38.8    47.0    42.5    39.3    37.2    38.2    52.4    53.3    52.8    51.9    42.9    47.0    52.1    45.9    48.8    52.3    43.7    47.6     45.4    45.3
        20.000  34.7    51.5    41.4    37.3    47.6    41.9    38.7    36.9    37.8    51.8    52.6    52.2    45.6    41.6    43.5    52.3    41.7    46.4    54.0    42.2    47.4     44.5    44.4
Confusion Matrix:
        neu     hap     sad     ang     fea     dis     sur
neu     3.75    0.74    1.39    0.27    0.27    0.57    0.74
hap     1.01    7.75    2.03    1.40    1.40    0.88    1.31
sad     2.46    1.70    7.18    0.76    1.80    1.39    0.76
ang     0.53    1.70    0.94    8.43    0.78    1.19    1.21
fea     1.21    2.09    2.17    1.23    7.28    0.86    0.59
dis     1.01    1.87    1.62    2.07    1.09    6.34    0.90
sur     0.90    2.58    1.19    1.70    0.74    1.25    7.00
Best macro f1 47.4
Best macro f1 std 3.7
Selected params: 1.00000

```

To test the file, the code script below is run: 
```python
# used trained model 
# to classify an unknown emotion
from pyAudioAnalysis import audioTrainTest as aT
files_to_test = [r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\002_happy.wav",
                 r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\002_disgust.wav",
                 r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\002_neutral.wav", 
                 r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\003_surprised.wav", 
                 r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\003_angry.wav", 
                 r"C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\003_neutral.wav"]
for f in files_to_test:
    print(f'{f}:')
    c, p, p_nam = aT.file_classification(f, "svm_emotions","svm")
    
    # print probabilities for each emotion
    for i in range(len(p_nam)):
        print(f'P({p_nam[i]} = {p[i]:.2f})')
    print()
```


It outputs: 

```
C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\002_happy.wav:
P(neutral = 0.02)
P(happy = 0.08)
P(sad = 0.56)
P(angry = 0.02)
P(fearful = 0.05)
P(disgust = 0.27)
P(surprised = 0.01)

C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\002_disgust.wav:
P(neutral = 0.05)
P(happy = 0.06)
P(sad = 0.80)
P(angry = 0.01)
P(fearful = 0.02)
P(disgust = 0.06)
P(surprised = 0.00)

C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\002_neutral.wav:
P(neutral = 0.01)
P(happy = 0.01)
P(sad = 0.68)
P(angry = 0.00)
P(fearful = 0.03)
P(disgust = 0.26)
P(surprised = 0.00)

C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\003_surprised.wav:
P(neutral = 0.05)
P(happy = 0.05)
P(sad = 0.85)
P(angry = 0.01)
P(fearful = 0.03)
P(disgust = 0.01)
P(surprised = 0.00)

C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\003_angry.wav:
P(neutral = 0.04)
P(happy = 0.04)
P(sad = 0.82)
P(angry = 0.01)
P(fearful = 0.06)
P(disgust = 0.03)
P(surprised = 0.00)

C:\Users\carol\OneDrive\Desktop\chao_py\emo_classification\test\003_neutral.wav:
P(neutral = 0.02)
P(happy = 0.04)
P(sad = 0.70)
P(angry = 0.01)
P(fearful = 0.04)
P(disgust = 0.19)
P(surprised = 0.01)
```

For some reason, the model predicts "sad" for each audio file. Some of the issues might be the following: 

1. When training the model, the following line of code was used: ```extract_features_and_train(dirs, mt, mt, st, st, "svm", "svm_emotions")```. From the pyAudioAnalysis documentation we see that this function has the following syntax: ```feature_extraction_train(listOfDirs, mtWin, mtStep, stWin, stStep, classifierType, modelName, computeBEAT)```. For ```classifierType```, ```svm_rbf``` is used for RBF kernel, while ```svm``` for linear kernel. In our intial training of the model, we used a linear kernel, which will not work for the multiclass classification. We will train another model using ```svm_rbf```, as it attempts to create more complex decision boundaries that can handle non-linear relationships in the data. This kernel is more flexible and can adapt to a wider variety of data structures (not linearly separable).
2. Preprocess data. 

### pyAudioAnalysis
Resources:     
https://dev.to/dolbyio/creating-audio-features-with-pyaudio-analysis-4mbp       
https://medium.com/behavioral-signals-ai/intro-to-audio-analysis-recognizing-sounds-using-machine-learning-20fd646a0ec5
## Week of 11/4/24
### Goals
1. Design experiement to standarize pitch values.
2. Write code script for pitch data analysis.


### Pitch Data Analysis Code
This code takes a series of wav audio files and creates a data frame that contains their pitch audio features. However, there are issues surroudning the ```f, f_names = ShortTermFeatures.feature_extraction(x, Fs, int(0.0055*Fs), int(0.00229*Fs))``` array size. ```feature_extraction``` uses the arguments window (short-term window size) and step (the short-term window step). Changing these values changes the array size, however, the array does not reshape correctly. 

Here is the directory map:   

C:\Users\carol\OneDrive\Desktop\chao_py\audio_data
│     
├── angry     
│   ├── 001_angry.wav      
│     
├── happy      
│   ├── 001_happy.wav     
│     
├── sad     
│   ├── 001_sad.wav     
│     
└── neutral      
    ├── 001_neutral.wav     

    
``` python
import os
import pandas as pd
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import numpy as np



# Function to extract pitch-related features from an audio file
def extract_pitch_features(audio_file):
    print(f"Processing file: {audio_file}")  # Debugging line
    try:
        [Fs, x] = audioBasicIO.read_audio_file(audio_file)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None

    # Extract short-term features
    f, f_names = ShortTermFeatures.feature_extraction(x, Fs, int(0.0055*Fs), int(0.00229*Fs))
    pitch_index = f_names.index('pitch')

    # Get pitch values
    pitch_values = f[pitch_index]

    # Calculate pitch-related features
    pitch_range = np.max(pitch_values) - np.min(pitch_values)
    slope = np.polyfit(np.arange(len(pitch_values)), pitch_values, 1)[0]
    mean_pitch = np.mean(pitch_values)
    min_pitch = np.min(pitch_values)
    max_pitch = np.max(pitch_values)
    pitch_variability = np.std(pitch_values)

    return {
        'mean_pitch': mean_pitch,
        'min_pitch': min_pitch,
        'max_pitch': max_pitch,
        'pitch_range': pitch_range,
        'pitch_slope': slope,
        'pitch_variability': pitch_variability
    }

# Function to collect all data
def collect_data(directory):
    data = []
    for emotion in os.listdir(directory):
        emotion_folder = os.path.join(directory, emotion)
        if os.path.isdir(emotion_folder):
            for audio_file in os.listdir(emotion_folder):
                audio_file_path = os.path.join(emotion_folder, audio_file)
                if audio_file_path.endswith('.wav'):
                    # Extract participantID and emotion from the filename
                    filename = os.path.basename(audio_file)
                    parts = filename.split('_')
                    if len(parts) == 2:  # Ensure the filename is correctly formatted (e.g., 001_angry.wav)
                        participant_id = parts[0]
                        emotion = parts[1].replace('.wav', '')  # Remove the .wav extension
                        
                        print(f"Found audio file: {audio_file_path} (Participant: {participant_id}, Emotion: {emotion})")
                        
                        # Extract pitch features
                        features = extract_pitch_features(audio_file_path)
                        
                        if features:  # Only add features if they were successfully extracted
                            features['emotion'] = emotion
                            features['participantID'] = participant_id
                            features['file_path'] = audio_file_path
                            data.append(features)

    # If data is collected, convert to DataFrame
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        print("No data was collected.")
        return pd.DataFrame()  # Return an empty DataFrame if no features were extracted

# Define the path to your recordings directory
recordings_directory = r"C:\Users\carol\OneDrive\Desktop\chao_py\audio_data"

# Collect data
df = collect_data(recordings_directory)

# If DataFrame is not empty, save it
if not df.empty:
    df.to_csv('emotion_pitch_data.csv', index=False)
    print("Data saved to emotion_pitch_data.csv")
else:
    print("No data was extracted, please check the directory and files.")

# Display the first few rows of the DataFrame
print(df.head())
```


### Standardizing Pitch Features
In order to accurately detect emotions based on pitch across all races, ages, and both sexes, we must find standard values for pitch and universal changes. The important features for emotion detection using pith are the following: mean pitch, min and max pitch, pitch range, pitch slope, and pitch variability. We will run an experiement with 10-15 diverse participants to analyze how these features change for each emotion. The participants will be asked to say the following statements in their normal tone (with a focus on the designated emotion): 

Neutral: "Today feels pretty ordinday; nothing really exciting is happening. I'm just going through my usual routine, trying to stay productive and make the most of it."     
Happy: "I'm over the moon about the promotion I just got at work! It’s such a great opportunity, and I can’t wait to take on new challenges and celebrate with friends."    
Sad: "I'm feeling really down today. I just received some upsetting news about a close friend, and it’s hard to shake off the weight of that sadness."     
Angry: "I'm incredibly frustrated with the way things are going right now. I can't believe you did that! Why can’t anyone see how frustrating this is? I’ve had it up to here."     


## Week of 10/28/24
### Goals 
1. Analyze pitch and MCFFs graphs for emotion set (happiness, sadness, anger, fear, surprise, disgust). Note major characteristics and compare to research findings.
2. Write emotion detection function using audio features based on analysis of pitch and MCFFs. 

### Emotion Detection using Pitch 
The analysis of audio features including pitch and MCFFs below has led to the conclusion that pitch is the easiest and most helpful data point to analyze in order to detect emotions. The following code script used the information collected during data analysis below and the Audio Features and Emotions table. 

``` python
import librosa
import numpy as np

def load_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)
    return y, sr

def extract_pitch(y, sr):
    # Estimate pitch using librosa's pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Extract pitch values
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Filter out zero values
            pitch_values.append(pitch)
    
    return np.array(pitch_values)

import numpy as np

def analyze_emotion(pitch_values):
    """
    Analyzes a list of pitch values to determine the associated emotion.

    Parameters:
        pitch_values (list or np.array): An array of pitch values.

    Returns:
        str: The inferred emotion based on pitch characteristics.
    """
    if len(pitch_values) == 0:
        return "No data"

    # Calculate pitch features
    mean_pitch = np.mean(pitch_values)
    pitch_range = np.ptp(pitch_values)  # Peak-to-peak range
    pitch_variation = np.std(pitch_values)  # Standard deviation for variability
    pitch_diff = np.diff(pitch_values)  # Difference between consecutive pitches

    # Emotion inference based on refined pitch characteristics
    if mean_pitch > 180 and pitch_variation > 20:  # High pitch and variability for happiness
        return "Happiness"
    elif mean_pitch > 180 and np.max(pitch_diff) > 50:  # High pitch and sharp rising for surprise
        return "Surprise"
    elif mean_pitch < 150 and pitch_variation < 15 and np.all(pitch_diff < 0):  # Low pitch, flat and falling for sadness
        return "Sadness"
    elif mean_pitch < 150 and pitch_variation < 10:  # Low pitch, flat for disgust
        return "Disgust"
    elif pitch_variation > 50 and (mean_pitch > 180 or np.max(pitch_diff) > 50):  # Sharp variation, high pitch for anger
        return "Anger"
    elif mean_pitch > 180 and (np.any(pitch_diff > 0) or pitch_variation > 20):  # High pitch, rising at end or shaky for fear
        return "Fear"
    else:
        return "Neutral/Uncertain"




def detect_emotion(file_path):
    # Load and process the audio file
    y, sr = load_audio(file_path)
    pitch_values = extract_pitch(y, sr)
    
    # Infer emotion from pitch data
    emotion = analyze_emotion(pitch_values)
    return emotion

# Example usage
file_path = r"C:\Users\cchao2869\Desktop\pyAudio\happy_ex.wav"
emotion = detect_emotion(file_path)
print(f"Detected Emotion: {emotion}")
```
### Pitch and MCFFs Graph Analysis

Using the code below, several audio files were tested where the speaker attempted to exemplify each of the emotions: happiness, sadness, anger, surprise, fear, and disgust. Data was extracted for the pitch and mel-frequency cepstral coefficients (MFCCs) of each audio file. MFCCs capture the timbral features of the audio, reflecting the texture and quality of the sound. Higher MFCC values may indicate brighter, richer sounds, while lower values might suggest dull or flat sounds. One limitation of the MCFF data is that it appears that the values are 0 for the majority of coefficients. Analysis of the pitch and MCFFs diagrams for each emotion is described below: 

#### Interpreting MCFFs Graphs
While the graphs for pitch are fairly straightforward, those for MCFFs can be more complicated to analyze. Some general guidelines are below to simplify the process: 

1. X-axis: Time in seconds, Y-axis: MFCC coefficient index (1 to 13)
2. Color: Red (High Values) - Higher amplitudes of the MFCC coefficients. This suggests that the specific timbral features captured by those coefficients are more pronounced in the audio signal during those time frames.     
Blue (Low Values) - Lower amplitudes of the MFCC coefficients, suggesting that those timbral features are less prominent or almost absent during those time frames.

#### Fear
Pitch: Fear is characterized by sharp rises in pitch. This is shown on the graph below at frame 325.    
MCFFs: The shaky, sharp audio quality displayed in fear is visible in the graph for MCFFs as well. As shown in the higher MCFFs (10-13), the timbre of the audio shifts quickly.  
![fear_ex](https://github.com/user-attachments/assets/3dfdbe37-66d8-4efe-843a-0a64b786e41b)

![fear_mcffs](https://github.com/user-attachments/assets/6304e4ba-39a2-4229-bf5e-54db47f3b4f3)

#### Happiness
Pitch: Happiness is characterized by a consitent shift from low to high. This is shown from fram 200 to 400. Note: audio began before speech, so data from (0, 2)U(4.5, 6) can be disregarded.      
MCFFs: From 2-4.5 seconds, a visible change is evident in the MCFFs graph. This is consistent with the shift in pitch during frames 200 to 400. 
![happy_pitch](https://github.com/user-attachments/assets/5215c441-15a7-41d5-a7eb-669e6fbef05f)

![happy_mcffs](https://github.com/user-attachments/assets/b6152792-25ef-48a8-8353-ed6ea4dfa60d)

#### Sadness
Pitch: Sadness is characterized by a low pitch with little variations. This is shown below because the X-axis (pitch in dB) ranges from 60 to 160, and is almost always at around 110 dB.      
MCFFs: The MCFFs data below shows consistently low MCFFs values during the interval of speech. This aligns with the characteristics of sadness as low, dull, and flat timbre. 
![sad_pitch](https://github.com/user-attachments/assets/e3eeb22b-1da1-4f18-b7f9-89df5886c961)

![sad_mcffs](https://github.com/user-attachments/assets/80e6febe-37fd-4657-81d8-3f5c5f5427e4)

#### Anger
Pitch: Anger is characterized by high variations in pitch. This is evident in the range of pitch from 50-250 frames.    
![anger_pitch](https://github.com/user-attachments/assets/5d273698-50e1-4e1c-8f7f-984248bc4495)
![anger_mcffs](https://github.com/user-attachments/assets/3025edbe-3fcf-4713-9300-91a487fc52d3)


#### Disgust 
![disgust_pitch](https://github.com/user-attachments/assets/6e1860f1-f889-4818-b1fe-58ac09063515)

![disgust_mcffs](https://github.com/user-attachments/assets/75cfb177-1fb8-4e53-9a41-b95fca5a90b7)


#### Surprise
![surprise_pitch](https://github.com/user-attachments/assets/2424454a-c079-4ed6-9c2a-0306e3c208cd)

![suprise_pitch](https://github.com/user-attachments/assets/cd68a514-a273-44c3-aac8-85243faf45b4)

## Week of 10/21/24
### Goals 
1. Research audio analysis libraries/models with Python and choose one that best fits program.
2. Define relationship between audio features and emotions.
3. Download and test audio analysis library/model.

----

### Audio Features and Emotions
We will increase the accuracy of the emotion detection program by extracting additional data points through audio analysis. According to [Detection and Analysis of Human Emotions through Voice and Speech Pattern Processing](https://arxiv.org/pdf/1710.10198), the most important audio features are pitch, SPL, timbre, and time gaps between consecutive words of speech. These relate to emotions as shown in the table below. 

| **Audio Feature**        | **Happiness**                                       | **Surprise**                                        | **Sadness**                                          | **Disgust**                                          | **Anger**                                             | **Fear**                                               |
|--------------------------|----------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|
| **Pitch**                 | High pitch, variable                               | High pitch, rising sharply                         | Low pitch, flat and falling                         | Low pitch, flat                                      | Sharp, high variation, sometimes high-pitched          | High pitch, rising at the end, or shaky                |
| **SPL (Loudness)**        | Moderate to high                                   | High                                                | Low                                                 | Low                                                  | High, sometimes very loud                              | Moderate to high, varies with intensity                |
| **Timbre**                | Bright, rich, clear tones                          | Sharp, clear tones                                 | Dull, flat, low energy                              | Harsh, rough, tense                                  | Harsh, strained, sharp                                | Soft or tense, sometimes breathy                       |
| **Time Gaps Between Words**| Rapid speech with shorter gaps                    | Rapid speech with shorter gaps                     | Slow speech, longer pauses between words            | Slow, deliberate speech                              | Fast, tense speech with short gaps, sometimes irregular| Moderate speech speed with irregular or long pauses    |

#### Explanation of Features:
- Pitch: The perceived frequency of the sound, typically higher in excited states (happiness, fear, surprise) and lower in subdued states (sadness, disgust).
- SPL (Sound Pressure Level): Related to the loudness of the speech. Higher SPL often indicates intense emotions like anger or surprise, while lower SPL is associated with sadness or calm emotions.
- Timbre: The quality or "color" of the voice, affected by harmonic content and how tense or relaxed the voice sounds. Emotions like happiness and surprise have richer harmonic content, while sadness and disgust often have dull or harsh qualities.
- Time Gaps Between Words: The pauses or silences in speech. Shorter gaps and faster speech are typically seen in emotions like happiness and anger, while longer gaps are associated with sadness or fear.


This information was found through studies in speech emotion recognition (SER). In our research in SER, we found that might be benficial in the future to our theraputic application in emotion detection by defining possible stimuli to emotions. See "Table of Speech and Emotions" on pages 47 and 48 of [Emotion recognition in human-computer interaction](https://www.researchgate.net/publication/3321357_Emotion_recognition_in_human-computer_interaction). 

| **Stimulus**                      | **Cognition**               | **Emotion**   | **Behavior**  |
|-----------------------------------|-----------------------------|---------------|---------------|
| **Threat**                        | Danger                      | Fear          | Escape        |
| **Obstacle**                      | Enemy                       | Anger         | Attack        |
| **Potential mate**                | Possess                     | Joy           | Mate          |
| **Loss of valued individual**     | Abandonment                 | Sadness       | Cry           |
| **Member of one’s group**         | Friend                      | Acceptance    | Groom         |
| **Unpalatable object**            | Poison                      | Disgust       | Vomit         |
| **New territory**                 | What’s out there?           | Expectation   | Map           |
| **Unexpected object**             | What is it?                 | Surprise      | Stop          |

### pyAudioAnalysis

pyAudioAnalysis is a library on Python for audio feature extraction, classification, segmentation. pyAudioAnalysis can extract data features such as pitch, intensity, and spectral properties, so we will use it in our emotion detection program. We will begin by simply analyzing pitch, SPL, and timbre, as these are the easiest audio features to extract. After downloading pyAudioAnalysis and the required dependencies (as well as numpy and librosa) shown below, we have the following code: 

#### Installation of libraries:
```pip install librosa soundfile numpy ```

#### Initial audio extraction code: 
``` python
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

# Load the audio file
audio_path = r"C:\Users\cchao2869\Desktop\pyAudio\chao_audio.wav"
y, sr = librosa.load(audio_path, sr=None)

# Extract Pitch using librosa's YIN (fundamental frequency estimation)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Handle NaN values in pitch
mean_f0 = np.nanmean(f0)
f0_cleaned = np.where(np.isnan(f0), mean_f0, f0)

# Calculate SPL (Sound Pressure Level)
rms = np.sqrt(np.mean(y**2))  # Root mean square for pressure level
p_ref = 20e-6  # Reference sound pressure in air
spl = 20 * np.log10(rms / p_ref)  # SPL in decibels

# Extract Timbre features (MFCCs and spectral features)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCC coefficients
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# Print results
print("Pitch (Hz):", f0_cleaned)
print("SPL (dB):", spl)
print("MFCCs (Timbre):", mfccs)
print("Spectral Centroid (Timbre):", spectral_centroid)
print("Spectral Bandwidth (Timbre):", spectral_bandwidth)
print("Spectral Contrast (Timbre):", spectral_contrast)

# Plotting Pitch
plt.figure(figsize=(10, 6))
plt.plot(f0_cleaned, label='Pitch (Hz)', color='blue')
plt.xlabel('Frame')
plt.ylabel('Pitch (Hz)')
plt.title('Extracted Pitch over Time')
plt.legend()
plt.grid()
plt.show()

# Plotting MFCCs
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCCs')
plt.show()
```


Output: ![Figure_1](https://github.com/user-attachments/assets/75eb4079-191a-4e21-a0f4-ea66a41d4761)



## Week of 10/14/24
### Goals 
1. Streamline the process of running command and Python script.
2. Increase accuracy of emotion detection.
3. Research decision trees and implement ```if then``` framework.

----
### Decision Trees / If-Then

```python
def ask_question(emotion):
    if emotion == "Happiness":
        return "I'm glad to hear it!"
    elif emotion in ["Sadness", "Anger"]:
        return "What about your situation makes you feel this way?"
    elif emotion == "Fear":
        return "Are you concerned for your family?"
    else:
        return "How is your family?"  # Default question for neutral or other emotions

```

### Increasing Accuracy 

Using the ```emotion_detection``` function below , we can increase the accuracy of the program by changing the following variables.     
- Use of AU intesity and binary data in unison
- Order of emotions
- Threshold of AU intensity data
- Delay in loop for webcam lag
- AUs for each emotion
- Amount of time each emotion is displayed
- Delay in emotional response after question is posed

Currently, the program has the hardest time detecting anger, disgust, and fear. Happiness is the easiest to detect. However, the program confuses emotions such as anger and disgust for happiness if the user is smiling slightly. Because of this, I am considering adding a counter argument that does not detect happiness if other AUs are present. The first change made to the emotion detection function was to change the order of the ```else if``` statements. This way, the program will look for anger, disgust, etc. before happiness. I am currently working on the use of AU intensity and binary data and which is more accurate.


``` python
def detect_emotion(data):
    if (data['AU04c'] == 1 and data['AU05c'] == 1 and data['AU07c'] == 1 and data['AU23c'] == 1):  # Anger
        return "Anger"
    elif (data['AU01c'] == 1 and data['AU04r'] > threshold and data['AU15c'] == 1):  # Sadness
        return "Sadness"
    elif (data['AU09c'] == 1 and data['AU15c'] == 1):  # Disgust
        return "Disgust"
    elif (data['AU06c'] == 1 and data['AU12c'] == 1):  # Happiness
        return "Happiness"
    elif (data['AU01c'] == 1 and data['AU02c'] == 1 and data['AU05c'] == 1 and data['AU26c'] == 1):  # Surprise
        return "Surprise"
    elif (data['AU01c'] == 1 and data['AU02c'] == 1 and data['AU04r'] > threshold and 
          data['AU05r'] > threshold and data['AU07r'] > threshold and data['AU20c'] == 1 and data['AU26c'] == 1):  # Fear
        return "Fear"
    
    
    return "Neutral"
```
### Emotion Detection Code (V1)
The code was finalized to detect emotions using the AUs defined in [AU-Emotion Combinations](https://github.com/Jpark27614/Emotion-Detection-Program/blob/main/DOC.md#action-unit-combinations-and-emotions).  If no emotion is detected, 'Neutral' will be printed. The biggest issue when extracting AU data is the format of the headers. After troubleshooting, it was determined that the whitespaces must be stripped in order for the code to run properly. This was key in correctly calling AU data headers (ex. 'AU01_c'). 


``` python
import time
import csv
import os

def detect_emotion(data):
    """Detects the emotion based on the provided Action Units."""
    if (data['AU06'] == 1 and data['AU12'] == 1):  # Happiness
        return "Happiness"
    elif (data['AU01'] == 1 and data['AU04'] == 1 and data['AU15'] == 1):  # Sadness
        return "Sadness"
    elif (data['AU01'] == 1 and data['AU02'] == 1 and data['AU05'] == 1 and data['AU26'] == 1):  # Surprise
        return "Surprise"
    elif (data['AU01'] == 1 and data['AU02'] == 1 and data['AU04'] == 1 and 
          data['AU05'] == 1 and data['AU07'] == 1 and data['AU20'] == 1 and data['AU26'] == 1):  # Fear
        return "Fear"
    elif (data['AU04'] == 1 and data['AU05'] == 1 and data['AU07'] == 1 and data['AU23'] == 1):  # Anger
        return "Anger"
    elif (data['AU09'] == 1 and data['AU15'] == 1):  # Disgust
        return "Disgust"
    
    return "Neutral"

def follow_csv(filename):
    # Open the CSV file in read mode
    with open(filename, 'r') as file:
        # Check if the file is empty
        if os.stat(filename).st_size == 0:
            print("The file is empty.")
            return

        # Attempt to read the header row
        try:
            header = next(csv.reader(file))
            print(f"Header found: {header}")  # Debug output
        except StopIteration:
            print("The file is empty or no header found.")
            return

        # Get indices for Action Units, ensuring to strip whitespace
        try:
            # Strip whitespace from header names
            header = [name.strip() for name in header]
            AU1_index = header.index('AU01_c')
            AU2_index = header.index('AU02_c')
            AU4_index = header.index('AU04_c')
            AU5_index = header.index('AU05_c')
            AU6_index = header.index('AU06_c')
            AU7_index = header.index('AU07_c')
            AU9_index = header.index('AU09_c')
            AU12_index = header.index('AU12_c')
            AU15_index = header.index('AU15_c')
            AU20_index = header.index('AU20_c')
            AU23_index = header.index('AU23_c')
            AU26_index = header.index('AU26_c')
            print(f"Indices found - AU01: {AU1_index}, AU02: {AU2_index}, AU04: {AU4_index}, AU05: {AU5_index}, AU06: {AU6_index}, AU07: {AU7_index}, AU09: {AU9_index}, AU12: {AU12_index}, AU15: {AU15_index}, AU20: {AU20_index}, AU23: {AU23_index}, AU26: {AU26_index}")  # Debug output
        except ValueError as e:
            print(f"Header error: {e}")
            return

        while True:
            # Read the new line
            new_line = file.readline()
            if new_line:
                # Debug output
                reader = csv.reader([new_line.strip()])
                for row in reader:
                    try:
                        data = {
                            'AU01': int(float(row[AU1_index].strip())),
                            'AU02': int(float(row[AU2_index].strip())),
                            'AU04': int(float(row[AU4_index].strip())),
                            'AU05': int(float(row[AU5_index].strip())),
                            'AU06': int(float(row[AU6_index].strip())),
                            'AU07': int(float(row[AU7_index].strip())),
                            'AU09': int(float(row[AU9_index].strip())),
                            'AU12': int(float(row[AU12_index].strip())),
                            'AU15': int(float(row[AU15_index].strip())),
                            'AU20': int(float(row[AU20_index].strip())),
                            'AU23': int(float(row[AU23_index].strip())),
                            'AU26': int(float(row[AU26_index].strip())),
                        }
                        emotion = detect_emotion(data)
                        print(f"Detected Emotion: {emotion}")
                    except ValueError as e:
                        print(f"Value error: {e}")
            else:
                time.sleep(0.5)

# Replace with the path to your actively updating CSV file
follow_csv(r'C:\Users\carol\OneDrive\Desktop\OpenFace\processed\chao_face.csv')
```

### Using batch file to integrate PowerShell and Python

In order to begin webcam input (on PowerShell) and run Python script to analyze output, a batch file was created. When opened, this file runs the OpenFace commands on PowerShell, delays 5 seconds, and runs the Python script. The output of the python script is visible on PowerShell. 

Batch file: 
```
@echo off
cd "C:\Users\carol\OneDrive\Desktop\OpenFace"
start /B FeatureExtraction.exe -aus -device 0 -out_dir "C:\Users\carol\OneDrive\Desktop\OpenFace\processed" -of "chao_face"
timeout /t 5
cd "C:\Users\carol\OneDrive\Desktop\chao_py"
python csv_monitor.py

```

```@echo off```: Hides the command being executed, making the output cleaner.       
```cd```: Changes the directory to where OpenFace files are located.      
```start /B```: Runs the FeatureExtraction.exe command in the background.        
```timeout /t 5```: Waits for 5 seconds to allow csv to have data.       

Here is the output on PowerShell when the batch file ```run_extraction.bat``` is opened:      

https://github.com/user-attachments/assets/c15e8b2f-180a-4b26-b618-b062943dfa9d


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

Code to constantly print new lines in csv file: 
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
follow_csv(r'C:\Users\cchao2869\Desktop\OpenFace\processed\chao_face.csv')
```

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

