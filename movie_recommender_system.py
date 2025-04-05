# movie_recommender.py

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and preprocess dataset
def load_and_prepare_data(path='movies.csv'):
    movies_data = pd.read_csv(path)

    # Add index if not exists
    if 'index' not in movies_data.columns:
        movies_data.reset_index(inplace=True)

    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = movies_data[selected_features].apply(lambda row: ' '.join(row), axis=1)

    return movies_data, combined_features

# Step 2: Build the model
def build_similarity_matrix(combined_features):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

# Step 3: Recommend similar movies
def recommend_movies(movie_name, movies_data, similarity):
    list_of_all_titles = movies_data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not close_matches:
        print("❌ No close matches found. Please check the movie name.")
        return

    close_match = close_matches[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_scores = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Top 10 movies similar to '{close_match}':\n")
    recommended = []
    i = 0
    for movie in sorted_similar_movies:
        index = movie[0]
        title = movies_data.iloc[index]['title']
        if title.lower() != close_match.lower():
            recommended.append(title)
            i += 1
        if i == 10:
            break
    for idx, title in enumerate(recommended, 1):
        print(f"{idx}. {title}")

# Step 4: Main function for real-time input
def main():
    print("📽️ Welcome to the Movie Recommender System!")
    movies_data, combined_features = load_and_prepare_data()
    similarity = build_similarity_matrix(combined_features)

    while True:
        movie_name = input("\n🎯 Enter a movie you like (or type 'exit' to quit): ").strip()
        if movie_name.lower() == 'exit':
            print("👋 Exiting Movie Recommender. See you again!")
            break
        recommend_movies(movie_name, movies_data, similarity)

# Run
if __name__ == '__main__':
    main()
