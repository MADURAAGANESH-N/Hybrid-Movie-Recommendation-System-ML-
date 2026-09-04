import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:

    def __init__(
        self,
        movies_path="data/processed/movies_clean.csv",
        sparse_path="models/user_movie_sparse.npz",
        movie_map_path="models/movie_to_index.pkl",
        reverse_map_path="models/index_to_movie.pkl"
    ):
        self.movies = pd.read_csv(movies_path)

        self.user_movie_sparse = load_npz(
            sparse_path
        )

        self.movie_to_index = joblib.load(
            movie_map_path
        )

        self.index_to_movie = joblib.load(
            reverse_map_path
        )

    def recommend(self, movie_id, num_recommendations=10):

        if movie_id not in self.movie_to_index:
            return pd.DataFrame()

        movie_index = self.movie_to_index[movie_id]

        scores = cosine_similarity(
            self.user_movie_sparse[:, movie_index].T,
            self.user_movie_sparse.T
        ).flatten()

        similar_indices = scores.argsort()[
            -(num_recommendations + 1):
        ][::-1]

        similar_indices = [
            i for i in similar_indices
            if i != movie_index
        ][:num_recommendations]

        recommended_ids = [
            self.index_to_movie[i]
            for i in similar_indices
        ]

        result = self.movies[
            self.movies["movieId"].isin(recommended_ids)
        ][["movieId", "title", "genres"]].copy()

        score_map = {
            self.index_to_movie[i]: round(scores[i], 3)
            for i in similar_indices
        }

        result["collaborative_score"] = (
            result["movieId"].map(score_map)
        )

        return result.sort_values(
            "collaborative_score",
            ascending=False
        )