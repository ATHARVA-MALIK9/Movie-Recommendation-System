"""
Movie Recommendation System
============================
Implements two recommendation techniques:
1. Content-Based Filtering  - recommends based on movie genres/features
2. Collaborative Filtering  - recommends based on similar users' ratings

Author  : [Your Name]
Project : Internship Project - Movie Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ─────────────────────────────────────────────
#  Sample Dataset
# ─────────────────────────────────────────────

MOVIES = pd.DataFrame({
    "movie_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "title": [
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "The Avengers",
        "Iron Man",
        "The Notebook",
        "Titanic",
        "The Pursuit of Happyness",
        "Forrest Gump",
        "Toy Story",
    ],
    "genres": [
        "Action Crime Drama",
        "Action Sci-Fi Thriller",
        "Sci-Fi Drama Adventure",
        "Action Sci-Fi Adventure",
        "Action Sci-Fi Adventure",
        "Romance Drama",
        "Romance Drama",
        "Drama Biography",
        "Drama Romance Comedy",
        "Animation Comedy Family",
    ],
})

# Rows = users, Columns = movies (0 means not rated)
RATINGS = pd.DataFrame({
    "user_id":         [1, 2, 3, 4, 5],
    "The Dark Knight": [5, 4, 0, 0, 2],
    "Inception":       [4, 5, 0, 1, 3],
    "Interstellar":    [4, 4, 0, 2, 3],
    "The Avengers":    [3, 4, 0, 0, 2],
    "Iron Man":        [4, 5, 1, 0, 3],
    "The Notebook":    [0, 0, 5, 4, 3],
    "Titanic":         [0, 1, 5, 5, 4],
    "The Pursuit of Happyness": [2, 0, 4, 3, 5],
    "Forrest Gump":    [3, 2, 4, 4, 5],
    "Toy Story":       [2, 1, 3, 4, 4],
})


# ─────────────────────────────────────────────
#  1. Content-Based Filtering
# ─────────────────────────────────────────────

class ContentBasedRecommender:
    """
    Recommends movies similar to a given movie
    based on genre similarity using TF-IDF + Cosine Similarity.
    """

    def __init__(self, movies_df: pd.DataFrame):
        self.movies = movies_df.copy()
        self._build_similarity_matrix()

    def _build_similarity_matrix(self):
        """Vectorize genres and compute cosine similarity between all movies."""
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies["genres"])
        self.sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # Map movie title → index for quick lookup
        self.title_to_idx = pd.Series(
            self.movies.index, index=self.movies["title"]
        )

    def recommend(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Return top_n movies most similar to movie_title.

        Parameters
        ----------
        movie_title : str  – exact title of a movie in the dataset
        top_n       : int  – number of recommendations

        Returns
        -------
        DataFrame with recommended movies and similarity scores
        """
        if movie_title not in self.title_to_idx:
            raise ValueError(
                f"'{movie_title}' not found. "
                f"Available: {list(self.movies['title'])}"
            )

        idx = self.title_to_idx[movie_title]
        scores = list(enumerate(self.sim_matrix[idx]))
        # Sort by similarity score, skip the movie itself (score = 1.0)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

        movie_indices = [i[0] for i in scores]
        result = self.movies.iloc[movie_indices][["title", "genres"]].copy()
        result["similarity_score"] = [round(s[1], 4) for s in scores]
        result.reset_index(drop=True, inplace=True)
        return result


# ─────────────────────────────────────────────
#  2. Collaborative Filtering
# ─────────────────────────────────────────────

class CollaborativeFilteringRecommender:
    """
    Recommends movies to a user based on what similar users liked.
    Uses User-Based Collaborative Filtering with Cosine Similarity.
    """

    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings = ratings_df.set_index("user_id")
        self._build_user_similarity()

    def _build_user_similarity(self):
        """Compute cosine similarity between every pair of users."""
        self.user_sim = pd.DataFrame(
            cosine_similarity(self.ratings),
            index=self.ratings.index,
            columns=self.ratings.index,
        )

    def recommend(self, user_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Return top_n movie recommendations for user_id.

        Parameters
        ----------
        user_id : int – must be in the ratings dataset
        top_n   : int – number of recommendations

        Returns
        -------
        DataFrame with recommended movies and predicted scores
        """
        if user_id not in self.ratings.index:
            raise ValueError(
                f"User {user_id} not found. "
                f"Available users: {list(self.ratings.index)}"
            )

        # Movies the user has NOT yet rated
        user_ratings = self.ratings.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()

        if not unrated_movies:
            return pd.DataFrame({"message": ["User has rated all movies!"]})

        # Similarity scores with all other users
        sim_scores = self.user_sim[user_id].drop(user_id)

        predicted_scores = {}
        for movie in unrated_movies:
            # Ratings given by other users for this movie
            other_ratings = self.ratings[movie].drop(user_id)
            # Weighted average: similar users' opinions matter more
            weighted_sum = np.dot(sim_scores, other_ratings)
            sim_sum = sim_scores[other_ratings > 0].sum()
            if sim_sum > 0:
                predicted_scores[movie] = round(weighted_sum / sim_sum, 4)

        result = (
            pd.DataFrame.from_dict(
                predicted_scores, orient="index", columns=["predicted_rating"]
            )
            .sort_values("predicted_rating", ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={"index": "movie_title"})
        )
        return result


# ─────────────────────────────────────────────
#  Demo / Quick Test
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 55)
    print("       MOVIE RECOMMENDATION SYSTEM DEMO")
    print("=" * 55)

    # ── Content-Based ──
    cb = ContentBasedRecommender(MOVIES)
    print("\n📽  Content-Based Filtering")
    print("   Query movie: 'Inception'")
    print("-" * 55)
    print(cb.recommend("Inception"))

    # ── Collaborative Filtering ──
    cf = CollaborativeFilteringRecommender(RATINGS)
    print("\n👥  Collaborative Filtering")
    print("   Recommendations for User 3")
    print("-" * 55)
    print(cf.recommend(user_id=3))

    print("\n✅  Done! Check out app.py for the interactive UI.\n")


if __name__ == "__main__":
    run_demo()
