import streamlit as st
import sys
import os

# Allow Python to find src
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.hybrid_recommender import HybridRecommender


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎬 Hybrid Movie Recommendation System")

st.write(
    "Get movie recommendations using "
    "content-based and collaborative filtering."
)


# -----------------------------
# Load recommender
# -----------------------------

@st.cache_resource
def load_recommender():
    return HybridRecommender()


recommender = load_recommender()


# -----------------------------
# Movie selection
# -----------------------------

movie_titles = sorted(
    recommender.movies["title"]
    .dropna()
    .unique()
)

selected_movie = st.selectbox(
    "Select a movie",
    movie_titles
)


# -----------------------------
# Number of recommendations
# -----------------------------

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# -----------------------------
# Recommendation button
# -----------------------------

if st.button("🎯 Recommend Movies"):

    with st.spinner("Finding similar movies..."):

        recommendations = recommender.recommend(
            selected_movie,
            num_recommendations
        )

    if recommendations.empty:

        st.error(
            "Sorry, recommendations could not be generated."
        )

    else:

        st.success(
            f"Recommendations for **{selected_movie}**"
        )

        for _, row in recommendations.iterrows():

            st.subheader(row["title"])

            st.write(
                f"Genres: {row['genres']}"
            )

            st.write(
                f"Content Score: "
                f"{row['content_score']:.3f}"
            )

            st.write(
                f"Collaborative Score: "
                f"{row['collaborative_score']:.3f}"
            )

            st.write(
                f"Hybrid Score: "
                f"{row['hybrid_score']:.3f}"
            )

            st.divider()