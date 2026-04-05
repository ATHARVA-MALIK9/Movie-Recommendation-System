"""
app.py  –  Interactive Web UI for the Movie Recommendation System
Run with:  streamlit run app.py
"""

import streamlit as st
from recommender import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    MOVIES,
    RATINGS,
)

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Recommendation System")
st.caption("Uses Content-Based & Collaborative Filtering")

st.divider()

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
method = st.sidebar.radio(
    "Choose Recommendation Method",
    ("Content-Based Filtering", "Collaborative Filtering"),
    help="Content-Based: based on movie genres | Collaborative: based on similar users",
)
top_n = st.sidebar.slider("Number of Recommendations", 1, 8, 5)

# ─────────────────────────────────────────────
#  Content-Based Tab
# ─────────────────────────────────────────────
if method == "Content-Based Filtering":
    st.subheader("📽️ Find Similar Movies")
    st.write(
        "Select a movie you like and we'll recommend similar movies "
        "based on their **genres**."
    )

    movie_list = MOVIES["title"].tolist()
    selected_movie = st.selectbox("🎥 Pick a movie you enjoyed:", movie_list)

    if st.button("Get Recommendations", type="primary"):
        cb = ContentBasedRecommender(MOVIES)
        results = cb.recommend(selected_movie, top_n=top_n)

        st.success(f"Top {top_n} movies similar to **{selected_movie}**:")
        for i, row in results.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 3, 1.5])
                col1.markdown(f"**{i+1}. {row['title']}**")
                col2.caption(row["genres"])
                col3.metric("Score", f"{row['similarity_score']:.2f}")
            st.divider()

# ─────────────────────────────────────────────
#  Collaborative Filtering Tab
# ─────────────────────────────────────────────
else:
    st.subheader("👥 Personalized Recommendations")
    st.write(
        "Select a user and we'll suggest movies based on what "
        "**similar users** have rated highly."
    )

    user_ids = RATINGS["user_id"].tolist()
    selected_user = st.selectbox("👤 Select a User ID:", user_ids)

    # Show the selected user's existing ratings
    with st.expander("📊 View this user's current ratings"):
        user_row = RATINGS[RATINGS["user_id"] == selected_user].drop(columns="user_id").T
        user_row.columns = ["Rating (0 = not rated)"]
        user_row = user_row[user_row["Rating (0 = not rated)"] > 0]
        st.dataframe(user_row, use_container_width=True)

    if st.button("Get Recommendations", type="primary"):
        cf = CollaborativeFilteringRecommender(RATINGS)
        results = cf.recommend(user_id=selected_user, top_n=top_n)

        st.success(f"Top {top_n} recommendations for **User {selected_user}**:")
        for i, row in results.iterrows():
            col1, col2 = st.columns([4, 1.5])
            col1.markdown(f"**{i+1}. {row['movie_title']}**")
            col2.metric("Predicted Rating", f"{row['predicted_rating']:.2f}")
            st.divider()

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown(
    "- **Content-Based**: Converts genres into TF-IDF vectors and finds "
    "movies with the highest cosine similarity.\n"
    "- **Collaborative**: Finds users with similar rating patterns and "
    "predicts scores for unseen movies using weighted averages."
)
