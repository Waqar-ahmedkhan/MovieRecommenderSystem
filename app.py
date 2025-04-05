from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the movie data
movies_data = pd.read_csv('movies.csv')
selected_features = ['genres','keywords','tagline','cast','director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data[selected_features].apply(lambda row: ' '.join(row), axis=1)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)
movies_data.reset_index(inplace=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return render_template('result.html', movie_name=movie_name, recommended_movies=["No match found"])

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    similarity_scores = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for movie in sorted_similar_movies[1:11]:
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommended_movies.append(title_from_index)

    return render_template('result.html', movie_name=close_match, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
