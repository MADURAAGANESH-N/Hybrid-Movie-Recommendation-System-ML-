import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:

    def __init__(
        self,
        movies_path="data/processed/movies_clean.csv",
        tfidf_path="models/tfidf_matrix.pkl",
        indices_path="models/movie_indices.pkl"
    ):
        self.movies = pd.read_csv(movies_path)
        self.tfidf_matrix = joblib.load(tfidf_path)
        self.movie_indices = joblib.load(indices_path)

    def recommend(self, movie_title, num_recommendations=10):

        if movie_title not in self.movie_indices:
            return pd.DataFrame()

        movie_index = self.movie_indices[movie_title]

        scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix
        ).flatten()

        similar_indices = scores.argsort()[
            -(num_recommendations + 1):
        ][::-1]

        similar_indices = [
            i for i in similar_indices
            if i != movie_index
        ][:num_recommendations]

        result = self.movies.iloc[
            similar_indices
        ][["movieId", "title", "genres"]].copy()

        result["content_score"] = [
            round(scores[i], 3)
            for i in similar_indices
        ]

        return result