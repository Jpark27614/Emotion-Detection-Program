# Emotion-Detection-Program

## Table of Contents
* [Schedule](#Schedule)
* [Research](#Research)
* [Application](#Application) 

## Schedule 

* Finish brainstorm and research- Weeks 3-4
* Research, install OpenFace, and test- Weeks 4-5
* Develop and finalize our application for emotion detection- Weeks 5-7
* Test and create code for the application and response- Weeks 7-15
* Document- Weeks 4-17

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
    * **Confusion Matrix:** A table layout used to visualize the performance of an algorithm. Each row of the matrix represents the instances in an actual class, and each column represents all instances that are correctly predicted. 
![Screenshot 2024-09-12 134119](https://github.com/user-attachments/assets/578fd0cc-0b9d-4889-bdbb-b7707920ad60)[Image](https://medium.com/@Coursesteach/building-a-real-time-emotion-detection-with-python-7fe6090a125d)

  * **Deployment:** How do we send this program out to the world? Ex. App, website, concept...

## Action Units  
An Action Unit is a measure of the facial muscle movements defined by the Facial Action Coding System (FACS). We take the data from the action units to see what emotion is displayed. For example, happiness is represented by raised cheeks and a pulled corner lip (6+12). 

![Screenshot 2024-09-19 140930](https://github.com/user-attachments/assets/8758da33-ee0e-4b29-94b2-0c7a45a55be4) [Image](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/)

For our research on Action units websites and articles were very useful. A good source for understanding what AU is on [Imotions](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/). Another website that was useful to us was [The Emotional Intelligence Agency](https://www.eiagroup.com/resources/facial-expressions/facial-action-coding-system-facs/). These sources cannot be credible without mentioning !
[Paul Ekman](https://www.paulekman.com/). Ekman is a huge name in the field of emotion detection, he discovered that some facial expressions of emotions are universal and co-discovered micro-expressions. His research has changed how we think about emotional expression and influenced Action Units. 

**Happiness:** 6 + 12 (Cheek Raiser + Lip Corner Pull) 

**Sadness:** 1 + 4 + 15 (Inner Brow Raiser + Brow Lowerer + Lip Corner Depressor 

**Suprise:** 1 + 2 + 5 + 26 (Inner Brow Raiser + Outer Brow Raiser + Upper Lid Raiser + Jaw Drop)

**Fear:** 1 + 2 + 4 + 5 + 7 + 20 + 26 (Inner Brow Raiser + Outer Brow Raiser + Brow Lowerer + Upper Lid Raiser + Tightener + Lip Strecher + Jaw Drop) 

**Anger:** 4 + 5 + 7 + 23 (Brow Lowerer + Upper Lid Raiser + Lid Tightener + Lip Tightener) 

**Disgust:** 9 + 15 + 16 (Nose Wrinkler + Lip Corner Depressor + Lower Lip Depressor)

## Application 
Initial Ideas:
* Aid for people with disabilities regarding emotional awareness
* deception 


