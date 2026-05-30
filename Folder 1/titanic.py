import pandas as pd
import numpy as np

#load dataset
df= pd.read_csv("task1\Titanic-Dataset.csv")

#Missing values
print(df.isnull().sum())

df["Age"].fillna(df["Age"].median(),inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

df.drop("Cabin",axis=1, inplace=True)

df["Sex"]=df["Sex"].map({"male":0 ,"female":1})

df["Embarked"]= df["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
})

print(df.head())

print(df.isnull().sum())

from sklearn.model_selection import train_test_split

#select input features
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

#select target column
y = df["Survived"]

#split dataset
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size= 0.2, random_state= 42)

#display shapes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#create model
model = LogisticRegression()

#Train model
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import  confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#print classification report
print(classification_report(y_test, y_pred))

#visualize confusion matrix
sns.heatmap(cm, annot=True, fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
