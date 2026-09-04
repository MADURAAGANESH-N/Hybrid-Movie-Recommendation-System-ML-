import pandas as pd
from .content_recommender import ContentRecommender
from .collaborative_recommender import CollaborativeRecommender


class HybridRecommender:

    def __init__(
        self,
        content_weight=0.6,
        collaborative_weight=0.4
    ):

        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        self.content = ContentRecommender()
        self.collaborative = CollaborativeRecommender()

        self.movies = self.content.movies

    def recommend(
        self,
        movie_title,
        num_recommendations=10
    ):

        # Find selected movie
        movie_rows = self.movies[
            self.movies["title"] == movie_title
        ]

        if movie_rows.empty:
            return pd.DataFrame()

        movie_id = movie_rows.iloc[0]["movieId"]

        # Content recommendations
        content_result = self.content.recommend(
            movie_title,
            num_recommendations=50
        )

        # Collaborative recommendations
        collaborative_result = self.collaborative.recommend(
            movie_id,
            num_recommendations=50
        )

        # Merge results
        result = pd.merge(
            content_result,
            collaborative_result,
            on=["movieId", "title", "genres"],
            how="outer"
        )

        result["content_score"] = (
            result["content_score"].fillna(0)
        )

        result["collaborative_score"] = (
            result["collaborative_score"].fillna(0)
        )

        # Hybrid score
        result["hybrid_score"] = (
            self.content_weight *
            result["content_score"]
            +
            self.collaborative_weight *
            result["collaborative_score"]
        )

        result = result.sort_values(
            "hybrid_score",
            ascending=False
        )

        return result[
            [
                "movieId",
                "title",
                "genres",
                "content_score",
                "collaborative_score",
                "hybrid_score"
            ]
        ].head(num_recommendations)