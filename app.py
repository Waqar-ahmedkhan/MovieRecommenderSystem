import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page configuration
st.set_page_config(page_title="Movie Recommender", layout="centered")

# Load the movie dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv('movies.csv')
    except FileNotFoundError:
        st.error("Error: 'movies.csv' not found. Please upload the dataset.")
        return pd.DataFrame()

movies_data = load_data()

if not movies_data.empty:
    # Select relevant features
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Fill missing values with empty strings
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combine all selected features into a single string
    combined_features = movies_data[selected_features].apply(lambda row: ' '.join(row), axis=1)

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Calculate cosine similarity between movies
    similarity = cosine_similarity(feature_vectors)

    # Function to recommend similar movies
    def recommend_movies(movie_name):
        movie_name = movie_name.lower()
        list_of_titles = movies_data['title'].tolist()
        close_matches = difflib.get_close_matches(movie_name, list_of_titles, n=1, cutoff=0.6)

        if close_matches:
            closest_match = close_matches[0]
            index = movies_data[movies_data['title'] == closest_match].index[0]
            similarity_scores = list(enumerate(similarity[index]))
            sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

            return [movies_data.iloc[i[0]]['title'] for i in sorted_movies], closest_match
        else:
            return [], None

    # Streamlit App UI
    st.title("🎬 Movie Recommender System")
    st.markdown("Type a movie name to get top 10 similar movie recommendations.")

    # Input field
    movie_input = st.text_input("Enter Movie Name:")

    # Show recommendations
    if movie_input:
        recommendations, matched_movie = recommend_movies(movie_input)

        if recommendations:
            st.success(f"Top 10 recommendations for **{matched_movie}**:")
            for i, title in enumerate(recommendations, start=1):
                st.write(f"{i}. {title}")
        else:
            st.warning("Sorry, no similar movies found. Try a different title.")

    # Upload option in case the file is missing
    if movies_data.empty:
        uploaded_file = st.file_uploader("Upload your 'movies.csv' file", type="csv")
        if uploaded_file is not None:
            movies_data = pd.read_csv(uploaded_file)
            st.experimental_rerun()
else:
    st.stop()
