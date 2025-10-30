# Movie Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols

# -----------------------------
# Read and import data
# -----------------------------

movieData = pd.read_csv('ratings.csv')

# View column names
print(movieData.columns)
print(movieData.columns.tolist())

# -----------------------------
# Ratings
# -----------------------------
my_rating = movieData['You rated']
imdb_rating = movieData['IMDb Rating']

# Combine into dataframe
df_rating = pd.DataFrame({
    'My rating': my_rating,
    'IMDb rating': imdb_rating
})

# Convert to long format (stack in R)
df_rating_stack = df_rating.melt(var_name='ind', value_name='values')

# Density plot
plt.figure(figsize=(8,5))
sns.kdeplot(data=df_rating_stack, x='values', hue='ind', fill=True, alpha=0.3)
plt.title("Density of My Rating vs IMDb Rating")
plt.show()

# -----------------------------
# Genre
# -----------------------------
genre = movieData['Genres'].astype(str)
genre[genre == ""] = "drama"

# Extract first genre
first_genre = genre.str.split(',').str[0]
unique_genre = first_genre.unique()

df_genre = pd.DataFrame({'first.genre': first_genre})

# Count per genre
count_genre = df_genre['first.genre'].value_counts().reindex(unique_genre)

# Bar plot of counts
plt.figure(figsize=(10,5))
sns.barplot(x=count_genre.index, y=count_genre.values)
plt.xticks(rotation=35, ha='right')
plt.ylabel("Count")
plt.xlabel("Genre")
plt.title("Number of Movies per Genre")
plt.show()

# -----------------------------
# Year
# -----------------------------
bins = [1930, 1960, 1990, 2016]
labels = ["1930-1960","1961-1990","1991-2016"]
years = pd.cut(movieData['Year'], bins=bins, labels=labels)

df_year = pd.DataFrame({'years': years})

# Bar plot of year bins
plt.figure(figsize=(8,5))
sns.countplot(x='years', data=df_year)
plt.xticks(rotation=35, ha='right')
plt.ylabel("Count")
plt.xlabel("Year Period")
plt.title("Number of Movies per Year Period")
plt.show()

# -----------------------------
# My rating and genres
# -----------------------------
df_rating_genre = pd.DataFrame({'rating': my_rating, 'genre': first_genre})

plt.figure(figsize=(10,5))
sns.kdeplot(data=df_rating_genre, x='rating', hue='genre', fill=True, alpha=0.3)
plt.title("Density of Ratings by Genre")
plt.show()

# -----------------------------
# Linear regression
# -----------------------------
# statsmodels requires formula interface for categorical variables
df_rating_genre['my_rating'] = my_rating
df_rating_genre['imdb_rating'] = imdb_rating

lm_fit = ols("my_rating ~ imdb_rating + C(genre)", data=df_rating_genre).fit()
print(lm_fit.summary())

fitted_val = lm_fit.fittedvalues

# -----------------------------
# Random Forest
# -----------------------------
# Encode first.genre as numeric for scikit-learn
le = LabelEncoder()
genre_encoded = le.fit_transform(first_genre)

rf_train_1 = pd.DataFrame({
    'imdb_rating': imdb_rating,
    'first_genre': genre_encoded
})

# Convert target to categorical
rf_label = my_rating.astype(str)

# Random forest classifier
rf = RandomForestClassifier(n_estimators=1000, random_state=1234)
rf.fit(rf_train_1, rf_label)

# Print feature importance
importances = rf.feature_importances_
for feature, importance in zip(rf_train_1.columns, importances):
    print(f"{feature}: {importance:.4f}")

# Optional: plot feature importance
plt.figure(figsize=(6,4))
sns.barplot(x=rf_train_1.columns, y=importances)
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()

# -----------------------------
# Mean rating by genre
# -----------------------------
mean_rating = df_rating_genre.groupby('genre')['rating'].mean().reindex(unique_genre)

plt.figure(figsize=(10,5))
sns.barplot(x=mean_rating.index, y=mean_rating.values)
plt.xticks(rotation=35, ha='right')
plt.ylabel("Mean Rating")
plt.xlabel("Genre")
plt.title("Mean Rating by Genre")
plt.show()