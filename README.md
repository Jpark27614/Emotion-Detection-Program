# Emotion-Detection-Program

## Table of Contents
* [Schedule](#Schedule)
* [Research](#Research)
* [Action Units](#Action_Units)
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

## Action_Units  
An Action Unit is a measure of the facial muscle movements defined by the Facial Action Coding System (FACS). We take the data from the action units to see what emotion is displayed. For example, happiness is represented by raised cheeks and a pulled corner lip (6+12). 

![Screenshot 2024-09-19 140930](https://github.com/user-attachments/assets/8758da33-ee0e-4b29-94b2-0c7a45a55be4) [Image](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/)

For our research on Action units websites and articles were very useful. A good source for understanding what AU is on [Imotions](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/). Another website that was useful to us was [The Emotional Intelligence Agency](https://www.eiagroup.com/resources/facial-expressions/facial-action-coding-system-facs/). These sources cannot be credible without mentioning !
[Paul Ekman](https://www.paulekman.com/). Ekman is a huge name in the field of emotion detection, he discovered that some facial expressions of emotions are universal and co-discovered micro-expressions. His research has changed how we think about emotional expression and influenced Action Units. 

### Action Unit Combinations and Emotions

*Hardest Action units to fake

**Happiness:** 6* + 12 (Cheek Raiser + Lip Corner Pull) 

**Sadness:** 1* + 4 + 15* (Inner Brow Raiser + Brow Lowerer + Lip Corner Depressor)

**Suprise:** 1 + 2 + 5 + 26 (Inner Brow Raiser + Outer Brow Raiser + Upper Lid Raiser + Jaw Drop)

**Fear:** 1* + 2* + 4* + 5 + 7 + 20 + 26 (Inner Brow Raiser + Outer Brow Raiser + Brow Lowerer + Upper Lid Raiser + Tightener + Lip Strecher + Jaw Drop) 

**Anger:** 4 + 5 + 7 + 23* (Brow Lowerer + Upper Lid Raiser + Lid Tightener + Lip Tightener) 

**Disgust:** 9 + 15 + 16 (Nose Wrinkler + Lip Corner Depressor + Lower Lip Depressor)

## Application 

We used this [website](https://www.gartner.com/smarterwithgartner/13-surprising-uses-for-emotion-ai-technology) to help us think of ideas for application. 

#### Brainstorm:
* Aid for people with disabilities regarding emotional awareness
* Detect deception/fabricated expressions. See types of facial expressions by Paul Eckman Group: https://www.paulekman.com/nonverbal-communication/types-of-facial-expressions/
* See how different products and adds spark emotions [website](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7264164/)
* Medical diagnosis (Ex. Depression, dementia, ADHD...)
* Video Games (game reacts to expression and changes based on it)
* **See what responses or questions evoke an emotional response then report what makes you sad, happy, scared, etc.]**
  *Another version of this idea could be used without a webcam. In place of a live video, the survey of questions could require a video recording response which would serve as the input for the facial recognition 

#### Final
We chose the application that asks questions to discover what people link emotions with. After lots of brainstorming, we ultimately landed on this choice because we wanted something that would help people personally/with their life matters. Although the idea of video games and medical diagnosis would be awesome, the emotional questionnaire deals more with mental health which is very closely linked with emotions and creates a more practical application for emotion detection. 

#### Example 
Emode: Hi, how do you feel about your family matters?

Human: Thanks for asking my family is great! * smiles *

Emode: Detects facial features of happiness and links family to happiness. 

## Webcam
To use the webcam for our code we had to take a couple of steps. First, we ran a line of code

First, we ran an OpenFace command with the webcam as a video input then stored the output to test.csv
We then uploaded the test.csv to a While True which sends it to the AU script, this makes it so it always checks for updates in the csv. 
