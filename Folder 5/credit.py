import pandas as pd #load and explore dataset
import numpy as np

df= pd.read_csv("task5\creditcard.csv")
df.head()
df.shape
df.columns
df.info()
df.isnull().sum()
df['Class'].value_counts()

fraud_percent= (df['Class'].value_counts()[1] / len(df))* 100
print(f"Fraud Transactions: {fraud_percent:.4f}%")

#data preprocessing and visualization
import matplotlib.pyplot as plt
import seaborn as sns

df['Class'].value_counts()

plt.figure(figsize=(6,4)) #plot class distribution
sns.countplot(x='Class', data=df)
plt.title("Distribution of Genuine vs Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,5)) #distribution of transaction amount
sns.histplot(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.show()

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
df['Amount']= scaler.fit_transform(df[['Amount']])
df['Time']= scaler.fit_transform(df[['Time']])
df.head()

#split dataset
X= df.drop('Class', axis=1)
y= df['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

#train logistic regression model
from sklearn.linear_model import LogisticRegression
model= LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred= model.predict(X_test)
print(y_pred[:10])

#evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
accuracy= accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

cm= confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap( cm , annot= True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

train_pred= model.predict(X_train) #check training accuracy

from sklearn.metrics import accuracy_score
print("Training Accuracy:", accuracy_score(y_train, train_pred))

test_pred= model.predict(X_test) #check testing accuracy
print("Testing Accuracy:", accuracy_score(y_test, test_pred))

sample= X_test.iloc[0].values.reshape(1,-1) #predict a single transaction
prediction= model.predict(sample)
print(prediction)

if prediction[0]==0: #dislay results
    print("Genuine Transaction")
else:
    print("Fraudulent Transaction")

print("Predicted:", prediction[0]) #comare with actual values
print("Actual:", y_test.iloc[0])