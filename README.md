# Emotion-Detection-Program

## Table of Contents
* [Description](#Description)
* [Research](#Research)
* [Weekly Overview](#Weekly_Overview)

### Problem: 
xxx

## Description
Our project is an emotion detection program using a webcam as an input. 

## Research 
Our first step was to comprehend the steps and process of facial recognition. We used the website [Medium.com](https://medium.com/@Coursesteach/building-a-real-time-emotion-detection-with-python-7fe6090a125d) to break down the process of facial recognition into manageable chunks. 
These are the steps we broke down the process into:
### 1. Data Collection and Prep
  * **Dataset:** A collection of photos assigned with different emotions to train the model on what the emotions look like.
    
    ![Variety-Facial-Emotion-Recognition-32-Data-Storage-The-dataset-used-in-this-research-is](https://github.com/user-attachments/assets/5415e327-81e1-4a14-8db0-6d4e9b958236)

  * **Preprocessing:** The process of resizing, scaling pixels, and augmentation for diversity (diverse images broaden capabilities of emotion detection so it detects more than a certain group of people Ex. All male images = only accurate for males)

    ![Screenshot 2024-09-10 132717](https://github.com/user-attachments/assets/8af09870-8e4b-4e6a-94a4-5fb886f86f63)

### 2. Model Building and Training 
  * We're planning to use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/README.md) for the base of our emotion detection model.
    * We then [installed](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation) OpenFace with the help of a [tutorial](https://www.youtube.com/watch?v=qknAAax8aCo).
  * **Training:** The process of feeding the model validated images to learn what different emotions look like. 

### 3. Video Detection 
  * **Preprocessing:** Extracting frames to analyze and process each frame to detect the emotion.
  * **Real-Time Detection:** Using frames from a webcam to process real-time emotions.

### 4. Evaluation and Detection 
  * **Model Evaluation:** Test and analyze the accuracy of the model, as well as the confusion matrix.
    * **Confusion Matrix:** A table layout used to visulaize performance of an algorithm. Each row of the matrix represents the instances in an actual class, and each column represents all instances that are correctly predicted. 
![Screenshot 2024-09-12 134119](https://github.com/user-attachments/assets/578fd0cc-0b9d-4889-bdbb-b7707920ad60)

  * **Deployment:** How do we send this program out to the world? Ex. App, website, concept...\

## Schedule 
* Finish brainstorm and research- Weeks 3-4
* Research, install OpenFace, and test- Weeks 4-5
* Develop and finalize our application for emotion detection- Weeks 5-7
* Test and create code for the application and response- Weeks 7-15
* Document- Weeks 4-17

## Weekly_Overview 

### Week 4 
* We installed OpenFace and tested out the program with videos. We also started documentation for our research and process on GitHub.

### Week 5
* 


