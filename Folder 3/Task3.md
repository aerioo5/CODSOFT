# Task 3 : Iris Flower Classification using Machine Learning

The Iris flower dataset consists of three species: setosa, versicolor, and virginica. These species can be distinguished based on their measurements. Now, imagine that you have the measurements of Iris flowers categorized by their respective species. Your objective is to train a machine learning model that can learn from these measurements and accurately classify the Iris flowers into their respective species.

## 📌 Project Overview
This project uses the famous Iris dataset to classify iris flowers into three species based on their sepal and petal measurements.
The three species are:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

A Random Forest Classifier is trained on the dataset to predict the species of a flower with high accuracy.

## 🎯 Objective
Build a machine learning model that can accurately classify iris flowers using:
  - Sepal Length
  - Sepal Width 
  - Petal Length 
  - Petal Width

## 📂 Dataset
The Iris dataset contains:
- 150 flower samples
- 4 numerical features
- 3 species classes

               Feature          Description
               Sepal Length     Length of sepal (cm)
               Sepal Width      Width of sepal (cm)
               Petal Length     Length of petal (cm)
               Petal Width      Width of petal (cm)
               Species          Flower species

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Random Forest Classifier


## 📊 Project Workflow
1. Data Loading:
   - Imported the Iris dataset using Pandas.
   - Checked dataset structure and dimensions.

2. Exploratory Data Analysis (EDA):
   - Checked for missing values.
   - Generated statistical summaries.
   - Visualized species distribution.
   - Analyzed feature relationships.

3. Data Preprocessing:
   - Separated features and target variable.
   - Split dataset into training and testing sets.

4. Model Training:
   - Trained a Random Forest Classifier on the training data.

5. Model Evaluation:
   - Calculated Accuracy Score.
   - Generated Confusion Matrix.
   - Generated Classification Report.

## 📈 Results

The Random Forest model achieved high classification accuracy on the test dataset.

Accuracy: 96% - 100%

The model successfully classified the three Iris flower species with excellent precision and recall.

## ▶️ How to Run

Navigate to the project directory: 
        
        cd iris-flower-classification

Install required packages: 
        
        pip install pandas matplotlib scikit-learn

Run the project: 
        
        python iris.py

## 📷 Output and Visualization 

<img width="640" height="480" alt="No  of Flowers in each species1" src="https://github.com/user-attachments/assets/248aa0ef-95e3-4a7e-b941-ca5ff8e483d7" />

<img width="640" height="480" alt="Petal Length vs Petal Width2" src="https://github.com/user-attachments/assets/bcb7eca0-873b-4e4a-982d-14607485e4a8" />

Accuracy: 1.0

Confusion Matrix:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

Classification Report.

## 📚 Learning Outcomes

- Data preprocessing using Pandas
- Exploratory Data Analysis (EDA)
- Train-Test Split
- Classification using Random Forest
- Model evaluation using Scikit-Learn
- Machine Learning workflow implementation

## 🔮 Future Improvements

- Test multiple ML algorithms.
- Improve accuracy with hyperparameter tuning.
- Build a web app using Streamlit.
- Save and deploy the trained model.
- Add real-time prediction features.

## Author

Developed as a beginner Machine Learning project using Python and Scikit-learn.

Developed by [Aditi]  
GitHub: [https://github.com/your-username](https://github.com/aerioo5)

⭐ If you like this project: Give this repository a ⭐ on GitHub!
