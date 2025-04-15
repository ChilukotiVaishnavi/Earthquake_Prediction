# 🌍 Earthquake Prediction Using Machine Learning 🌍

A web-based earthquake prediction system that uses **Machine Learning** to estimate earthquake intensity based on **latitude, longitude, and height**. Built using **Flask, Scikit-learn,Xgboost,AdaBoostClassifier,LinearRegression,SVM and Random Forest Classifier**.

## 📌 Features
-✅ Predicts earthquake intensity (Richter scale) using machine learning models
-✅ User-friendly web interface built with Flask and HTML/CSS
-✅ Utilizes multiple models:Random Forest Classifier,AdaBoost Classifier,Support Vector Machine (SVM),XGBoost Classifier & Regressor,Linear Regression
-✅ Shows output from the most accurate model (XGBoost Regressor)
-✅ Dynamically maps predicted values to risk levels
-✅ Validates and accepts both integers and floats as inputs
-✅ Automatically handles invalid inputs gracefully
-✅ Provides interactive messages with risk interpretation
-✅ Clean, mobile-friendly UI with alert messages and tooltips
-✅ Fully functional backend for future model comparison or selection

## 🛠️ Tech Stack
- 🔹 **Frontend:** HTML, CSS  
- 🔹 **Backend:** Flask (Python)  
- 🔹 **Machine Learning:** Random Forest Classifier & Regressor,AdaBoost Classifier,Support Vector Machine (SVM),XGBoost Classifier & Regressor,Linear Regression
- 🔹 **Data Handling:** NumPy, Pandas  
- 🔹 **Deployment:** Flask Server

## 🖥️ How to Run Locally
1️⃣ Install Python in your laptop or pc
2️⃣ Create a virtual environment
- Add python path to your environmental variables(In Properties --> Advanced System Settings --> Edit PATH variable in system variables)
- python -m venv env
3️⃣ Install dependencies
- py -m pip
- python.exe -m pip install --upgrade pip
- pip install -r requirements.txt
- pip install flask,numpy,pandas,scikit-learn,joblib,xgboost 
4️⃣ Run the Detector to download pkl files
-python detector.py
5️⃣ Run the Flask application
-python app.py
6️⃣ Open your browser and visit http://127.0.0.1:5000 or follow the link in terminal

## ✨ Output
![Image](https://github.com/user-attachments/assets/3d8bde0d-046f-4ee4-b071-b0c52fc00fd1)
![Image](https://github.com/user-attachments/assets/e4911858-56ed-45d1-b351-33417811707a)

