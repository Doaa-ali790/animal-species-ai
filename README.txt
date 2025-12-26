# 🐾 Animal Species AI

A computer vision and machine learning project that classifies animal images into 10 categories, displays related information, and converts the result into speech (text-to-speech).

---

## 🔍 Project Overview
This system takes an input image of an animal, predicts its class, and then:
1. Displays the animal name
2. Provides information about the animal
3. Converts the information into audible speech

The classification model was trained using a labeled animal image dataset.

---

## 🐾 Supported Species
The model recognizes the following animal classes:
- Dog  
- Cat  
- Horse  
- Cow  
- Sheep  
- Elephant  
- Butterfly  
- Chicken  
- Spider  
- Squirrel

---

## 📂 Dataset (Animals10)
The project uses the **Animals10** dataset, which contains thousands of labeled animal images across 10 categories.

Due to large file size, the dataset is **not included in this GitHub repository**.

### 📥 Download the Dataset
You can download the dataset from Google Drive:

🔗 **Google Drive Link:**  
https://drive.google.com/drive/folders/1ZvFS9homc612RQ9AlWEZtF6Q3ILgrQBc?usp=drive_link

After downloading:
1. Extract the ZIP file
2. Place the extracted folder in your project folder as:


dataset/


---

## 🧠 Model Used
- Feature extraction using **MobileNet**
- Classification using **Support Vector Machine (SVM)**
- Final model file:  


svc_pipeline_mobilenet.joblib


---

## 📦 Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt

🚀 Run the Project
🧪 Interactive Python Script
python app_interactive.py

🌐 Streamlit Web App
streamlit run app_streamlit.py

🔊 Text-to-Speech

Prerecorded audio files for each animal class are stored in the:

sounds/


folder. The appropriate sound file will play automatically based on the predicted class.

📌 Notes

This repository contains code and model files only.

Dataset must be downloaded separately via the Google Drive link above.

Best suited for educational and academic use.

Can be extended to API or deployed as a web application.

👩‍💻 Developer

Doaa Ali
