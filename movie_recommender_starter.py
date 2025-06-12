import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

df = pd.read_csv('movie_dataset.csv')

features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
	df[feature] = df[feature].fillna("")

def combine_features(row):
	return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
	

df["combined_features"] = df.apply(combine_features, axis = 1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
similarity_scores = cosine_similarity(count_matrix)


movie_user_likes = "Avatar"

index = get_index_from_title(movie_user_likes)

movie_row = list(enumerate(similarity_scores[index]))
descending_similarity = sorted(movie_row, key = lambda x:x[1], reverse=True)
descending_similarity = descending_similarity[1:]
for i in range(50):
	print(get_title_from_index(descending_similarity[i][0]))
