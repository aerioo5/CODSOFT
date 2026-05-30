import pandas as pd

df = pd.read_csv("task3\IRIS.csv") #load dataset

print(df.head()) #check info
print(df.info())
print(df.shape)

print(df.isnull().sum()) #EDA
print(df.describe())
print(df['species'].value_counts())

#visualize 
import matplotlib.pyplot as plt 

df['species'].value_counts().plot(kind='bar') 
plt.title("No. of Flowers in each species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

plt.scatter(df['petal_length'], df['petal_width'])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Lenth vs Petal Width")
plt.show()

#data preprocessing and trasin-test split 
from sklearn.model_selection import train_test_split

X= df.drop('species', axis=1) 
y= df['species']
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size= 0.2, random_state= 42
) #split dataset

print("Training Features:", X_train.shape) #check shapes
print("Testing Features:", X_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred= model.predict(X_test)

print("Model Trined Successfully!")
print("Predictions:", y_pred[:5])

#Evaluate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm= confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

report= classification_report(y_test, y_pred)
print("\nClassification Report:\n\n", report)
