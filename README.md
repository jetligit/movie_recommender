🎬 Movie Recommender

A content-based movie recommendation system built with Python that suggests movies similar to a given title by analyzing features like keywords, cast, genres, and director.

🌟 Features
Provides the top 10 similar movies based on content similarity.
Combines movie keywords, cast, genres, and director to compute similarity.
Uses cosine similarity on a Bag-of-Words representation of combined features.
Handles missing data gracefully by filling empty fields.

🧠 How It Works
Feature Combination: Each movie’s keywords, cast, genres, and director are merged into a single string.
Vectorization: The combined features are converted into a CountVectorizer matrix.
Similarity Calculation: Cosine similarity is computed between all movies.
Recommendation: Returns a ranked list of movies most similar to the input title.

🛠️ Technologies Used
Python 
pandas 
numpy 
scikit-learn – for CountVectorizer and cosine_similarity
