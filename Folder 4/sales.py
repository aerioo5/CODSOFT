import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"task4\advertising.csv")
print (df.head())
print (df.info())

#data exploration and visualization
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(12,3))

plt.subplot(1,3,1)
plt.scatter(df['TV'], df['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs Sales')

plt.subplot(1,3,2)
plt.scatter(df['Radio'], df['Sales'])
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.title('Radio vs Sales')

plt.subplot(1,3,3)
plt.scatter(df['Newspaper'], df['Sales'])
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.title('Newspaper vs Sales')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(),annot= True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#feature selection and train-test slit
X= df[['TV','Radio','Newspaper']]
y= df['Sales']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape )
print("X_test shape:", X_test.shape )
print("y_train shape:", y_train.shape )
print("y_test shape:", y_test.shape )

# train linear regression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_ )
print("Coefficients:", model.coef_ )

y_pred= model.predict(X_test)
print(y_pred[:5])

#model evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae= mean_absolute_error(y_test,y_pred)
mse= mean_squared_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)
print("MAE:",mae)
print("MSE:",mse)
print("R^2 Score:",r2)

#comare and actual pred values
comparison= pd.DataFrame ({
    'Actual Sales': y_test,
    'Predicted Sales': y_pred
})
print(comparison.head(10))

plt.figure(figsize=(8,5))
plt.plot(range(len(y_test)), y_test.values, marker='o',label='Actual')
plt.plot(range(len(y_pred)), y_pred, marker='x',label='Predicted')

plt.title('Actual vs Predicted Sales')
plt.xlabel('Test Samples')
plt.ylabel('Sales')
plt.legend()
plt.show()