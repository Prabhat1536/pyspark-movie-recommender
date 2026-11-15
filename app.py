import streamlit as st
import plotly.express as px
from src.backend import (
    load_data_and_model,
    get_bda_overview,
    get_user_dashboard_data,
)

st.set_page_config(
    page_title="PySpark ALS Movie Recommender | BDA Project",
    page_icon="🎬",
    layout="wide",
)

@st.cache_resource
def init_engine():
    return load_data_and_model()

movies_df, ratings_df, user_factors, item_factors = init_engine()
bda_stats = get_bda_overview(movies_df, ratings_df)

# --- Sidebar Controls ---
st.sidebar.title("🎛️ Recommender Controls")
all_user_ids = sorted(user_factors["id"].unique().tolist())
selected_user = st.sidebar.selectbox(
    "Select User ID (1 - 610)",
    options=all_user_ids,
    index=all_user_ids.index(42) if 42 in all_user_ids else 0,
)
top_n = st.sidebar.slider("Number of Recommendations (Top-N)", min_value=5, max_value=20, value=10)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ PySpark Model Specs")
st.sidebar.info(
    "**Algorithm:** ALS Matrix Factorization\n\n"
    "**Latent Factors (Rank):** 20\n\n"
    "**Regularization (λ):** 0.1\n\n"
    "**Max Iterations:** 15\n\n"
    "**Test Set RMSE:** 0.8735"
)

# --- Main Header ---
st.title("🎬 Distributed Movie Recommendation Engine (PySpark ALS)")
st.caption("Big Data Analytics (BDA) Project • Collaborative Filtering on MovieLens (100K+ Interactions)")

tab1, tab2, tab3 = st.tabs([
    "🎯 User Recommendations",
    "📊 BDA Dataset Analytics",
    "🧠 Model & Spark Architecture",
])

# ==========================================
# TAB 1: PERSONALIZED RECOMMENDATIONS
# ==========================================
with tab1:
    history_df, user_genre_stats, recs_df = get_user_dashboard_data(
        selected_user, top_n, movies_df, ratings_df, user_factors, item_factors
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active User ID", f"#{selected_user}")
    c2.metric("Movies Rated by User", len(history_df))
    c3.metric("User's Avg Rating", f"{history_df['rating'].mean():.2f} ⭐" if not history_df.empty else "N/A")
    c4.metric(
        "Top Predicted Score",
        f"{recs_df['Predicted_Rating'].max():.2f} ⭐" if not recs_df.empty else "N/A",
    )

    st.markdown("---")
    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        st.subheader(f"🔥 Top {top_n} Recommended Unseen Movies for User #{selected_user}")
        st.dataframe(
            recs_df[["movieId", "title", "genres", "Predicted_Rating", "Raw_ALS_Score"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        st.subheader("📈 Predicted Rating Comparison")
        fig_recs = px.bar(
            recs_df.sort_values("Raw_ALS_Score", ascending=True),
            x="Raw_ALS_Score",
            y="title",
            orientation="h",
            color="Predicted_Rating",
            color_continuous_scale="Viridis",
            labels={"Raw_ALS_Score": "ALS Affinity Score", "title": "Movie"},
        )
        fig_recs.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_recs, use_container_width=True)

    st.markdown("---")
    col_hist, col_genre = st.columns([1.15, 0.85])

    with col_hist:
        st.subheader(f"📜 User #{selected_user}'s Highest Rated Watch History")
        st.dataframe(history_df.head(10), use_container_width=True, hide_index=True)

    with col_genre:
        st.subheader(f"🎭 User #{selected_user}'s Genre Taste Profile")
        fig_user_genre = px.bar(
            user_genre_stats,
            x="genre",
            y="Movies_Watched",
            color="Avg_Rating",
            text="Avg_Rating",
            color_continuous_scale="Blues",
            labels={"genre": "Genre", "Movies_Watched": "Movies Watched"},
        )
        fig_user_genre.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_user_genre, use_container_width=True)

# ==========================================
# TAB 2: BDA DATASET ANALYTICS
# ==========================================
with tab2:
    st.subheader("Big Data Exploratory & Sparsity Analysis")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Interactions", f"{bda_stats['total_ratings']:,}")
    m2.metric("Unique Users", f"{bda_stats['num_users']:,}")
    m3.metric("Unique Movies", f"{bda_stats['num_movies']:,}")
    m4.metric("Matrix Sparsity", f"{bda_stats['sparsity']}%")
    m5.metric("Global Avg Rating", f"{bda_stats['avg_rating']} ⭐")

    st.markdown("---")
    g1, g2 = st.columns(2)

    with g1:
        st.subheader("⭐ Global Rating Distribution")
        fig_dist = px.bar(
            bda_stats["rating_counts"],
            x="Rating",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Teal",
        )
        fig_dist.update_layout(height=380)
        st.plotly_chart(fig_dist, use_container_width=True)

    with g2:
        st.subheader("🎞️ Top 10 Movie Genres by Rating Volume")
        fig_genres = px.pie(
            bda_stats["top_genres"],
            names="Genre",
            values="Ratings_Count",
            hole=0.45,
        )
        fig_genres.update_layout(height=380)
        st.plotly_chart(fig_genres, use_container_width=True)

    st.subheader("📊 Long-Tail User Activity Distribution (Ratings per User)")
    fig_longtail = px.histogram(
        bda_stats["user_activity"],
        x="Ratings_Count",
        nbins=40,
        labels={"Ratings_Count": "Number of Movies Rated by a User"},
        color_discrete_sequence=["#636EFA"],
    )
    fig_longtail.update_layout(height=340, yaxis_title="Number of Users")
    st.plotly_chart(fig_longtail, use_container_width=True)

# ==========================================
# TAB 3: SPARK & ALS ARCHITECTURE
# ==========================================
with tab3:
    st.subheader("Distributed Processing & ALS Matrix Factorization Pipeline")
    st.markdown(
        """
        ### Why Apache Spark (PySpark) for Recommendation Systems?
        1. **Handling 98.30% Matrix Sparsity:** A $610 \\times 9,724$ user-item interaction matrix contains **5.93 million** possible cells, of which only **100,836** are populated. Standard dense matrix operations waste memory; Spark distributes sparse RDD/DataFrame blocks across worker partitions.
        2. **Alternating Least Squares (ALS):** Decomposes the sparse rating matrix $R$ into two low-rank dense matrices:
           - **User Latent Matrix ($U$):** $610 \\times 20$
           - **Item Latent Matrix ($V$):** $9,724 \\times 20$
        3. **Hyperparameter Tuning (`CrossValidator`):** Evaluated across folds using Root Mean Square Error (**RMSE = 0.8735** on a 20% hold-out test split).
        4. **Decoupled Serving Architecture:** PySpark performs offline distributed training and exports Parquet latent factor tables (`user_factors.parquet`, `item_factors.parquet`), enabling low-latency real-time inference in Streamlit.
        """
    )