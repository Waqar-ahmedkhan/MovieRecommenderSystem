import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies_data = pd.read_csv('movies.csv')

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features into one
combined_features = movies_data[selected_features].apply(lambda row: ' '.join(row), axis=1)

# Convert text to numerical values using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate similarity
similarity = cosine_similarity(feature_vectors)


# Function to recommend movies based on user input
def recommend_movies(movie_name):
    # Find closest movie name from the dataset
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        movie_index = movies_data[movies_data['title'] == find_close_match[0]].index[0]
        similarity_scores = list(enumerate(similarity[movie_index]))
        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for movie in sorted_similar_movies[1:11]:  # Skip the first one as it will be the same movie
            movie_title = movies_data.iloc[movie[0]]['title']
            recommended_movies.append(movie_title)

        return recommended_movies
    else:
        return []


# Streamlit UI
st.title('Movie Recommender System')
st.write('Enter a movie name to get similar movie recommendations.')

# User input
movie_name = st.text_input('Enter Movie Name')

if movie_name:
    recommended_movies = recommend_movies(movie_name)

    if recommended_movies:
        st.write(f"Top 10 recommendations for '{movie_name}':")
        for i, movie in enumerate(recommended_movies):
            st.write(f"{i + 1}. {movie}")
    else:
        st.write("Sorry, no recommendations found.")
