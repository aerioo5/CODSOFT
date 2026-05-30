import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"task2\IMDb Movies India.csv", encoding='latin1')

print(df.head()
,df.shape
,df.columns
,df.info()
,df.describe())

#missin values
print(df.isnull().sum())

#remove null rows 
df=df.dropna()

#clean year
df['Year']=df['Year'].str.replace('(','')
df['Year']=df['Year'].str.replace(')','')
df['Year']=df['Year'].astype(int)

#clean duration
df['Duration']=df['Duration'].str.replace('min','')
df['Duration']=df['Duration'].astype(int)

#clean votes
df['Votes']=df['Votes'].str.replace(',','')
df['Votes']=df['Votes'].astype(int)

#convert rating
df['Rating']=df['Rating'].astype(float)

print(df.dtypes) #check datatypes
print(df.head()) #preview cleaned data

#top 10 highest rated movies
top_movies = df[['Name','Rating']].sort_values(by='Rating', ascending=False).head(10)
print(top_movies)

#plot to rated movies
plt.figure(figsize=(10,5))
sns.barplot(x='Rating', y='Name', data=top_movies)
plt.title("Top 10 Highest Rated Movies")
plt.show()

#no. of movies released per year
movies_per_year= df['Year'].value_counts().sort_index()
print(movies_per_year)

#plot movies released per year
plt.figure(figsize=(12,6))
movies_per_year.plot()
plt.title("Movies Released Per Year")
plt.xlabel("Year")
plt.ylabel("No. of Movies")
plt.show()

#Rating distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'],bins=20)
plt.title("Rating Distribution")
plt.show()

#does duration affect rating?
plt.figure(figsize=(8,5))
sns.scatterplot(x='Duration',y='Rating',data=df)
plt.title("Duration vs Rating")
plt.show()

#most popular genres
genre_count= df['Genre'].value_counts().head(10)
print(genre_count)

#plot top genres
plt.figure(figsize=(10,5))
sns.barplot(x=genre_count.values, y=genre_count.index)
plt.title("Top 10 Genres")
plt.show()

#directors with most movies
top_directors= df['Director'].value_counts().head(10)
print(top_directors)

#plot to directors
plt.figure(figsize=(10,5))
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title("Top Directors")
plt.show()

#correlation between numeric columns
correlation= df[['Year','Duration','Votes','Rating']].corr()
print(correlation)

#heatmap
plt.figure(figsize=(8,5))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()

#model training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

movie_df = df.copy()
movie_df = movie_df[['Genre','Director','Actor 1','Actor 2','Actor 3','Votes','Duration','Year','Rating']]

encoder = LabelEncoder()
movie_df['Genre'] = encoder.fit_transform(movie_df['Genre'])
movie_df['Director'] = encoder.fit_transform(movie_df['Director'])
movie_df['Actor 1'] = encoder.fit_transform(movie_df['Actor 1'])
movie_df['Actor 2'] = encoder.fit_transform(movie_df['Actor 2'])
movie_df['Actor 3'] = encoder.fit_transform(movie_df['Actor 3'])
X = movie_df.drop('Rating', axis=1)
y = movie_df['Rating']

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
model = LinearRegression() #linear regression
model.fit(X_train, y_train)

y_pred = model.predict(X_test) #predict ratings

mae = mean_absolute_error(y_test,y_pred)
print("MAE:",mae)
mse = mean_squared_error(y_test,y_pred)
print("MSE:",mse)
r2 = r2_score(y_test, y_pred)
print("R2 Score:",r2)

results = pd.DataFrame({'Actual Rating':y_test, 'Predicted Rating':y_pred}) #compare actual vs predicted rating
print(results.head(10))

plt.figure(figsize=(8,5)) #visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()
