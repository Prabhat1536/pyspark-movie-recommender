import os
import numpy as np
import pandas as pd
from src.config import DATA_DIR, DATASET_FOLDER, MODEL_DIR

MOVIES_PATH = os.path.join(DATA_DIR, DATASET_FOLDER, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, DATASET_FOLDER, "ratings.csv")
USER_FACTORS_PATH = os.path.join(MODEL_DIR, "user_factors.parquet")
ITEM_FACTORS_PATH = os.path.join(MODEL_DIR, "item_factors.parquet")


def load_data_and_model():
    """Loads raw MovieLens data and PySpark-trained ALS latent factor matrices."""
    movies_df = pd.read_csv(MOVIES_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    user_factors = pd.read_parquet(USER_FACTORS_PATH)
    item_factors = pd.read_parquet(ITEM_FACTORS_PATH)
    return movies_df, ratings_df, user_factors, item_factors


def get_bda_overview(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """Computes Big Data Analytics summary metrics and distributions."""
    num_users = ratings_df["userId"].nunique()
    num_movies = ratings_df["movieId"].nunique()
    total_ratings = len(ratings_df)
    sparsity = (1.0 - (total_ratings / (num_users * num_movies))) * 100

    # Rating value distribution (0.5 to 5.0)
    rating_counts = (
        ratings_df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    rating_counts.columns = ["Rating", "Count"]

    # Top 10 genres across all rated movies
    merged = ratings_df.merge(movies_df, on="movieId", how="inner")
    genres_series = (
        merged["genres"]
        .str.split("|")
        .explode()
    )
    genres_series = genres_series[genres_series != "(no genres listed)"]
    top_genres = genres_series.value_counts().head(10).reset_index()
    top_genres.columns = ["Genre", "Ratings_Count"]

    # Long-tail activity: ratings per user
    user_activity = (
        ratings_df.groupby("userId")
        .size()
        .reset_index(name="Ratings_Count")
    )

    return {
        "total_ratings": total_ratings,
        "num_users": num_users,
        "num_movies": num_movies,
        "sparsity": round(sparsity, 2),
        "avg_rating": round(ratings_df["rating"].mean(), 2),
        "rating_counts": rating_counts,
        "top_genres": top_genres,
        "user_activity": user_activity,
    }


def get_user_dashboard_data(
    user_id: int,
    top_n: int,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    user_factors: pd.DataFrame,
    item_factors: pd.DataFrame,
):
    """Fetches watch history, genre preferences, and ALS recommendations for a user."""
    user_ratings = ratings_df[ratings_df["userId"] == user_id].merge(
        movies_df, on="movieId", how="inner"
    )
    history_df = (
        user_ratings[["movieId", "title", "genres", "rating"]]
        .sort_values(by=["rating", "title"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # User's favorite genres weighted by rating
    genre_pref = (
        user_ratings.assign(genre=user_ratings["genres"].str.split("|"))
        .explode("genre")
    )
    genre_pref = genre_pref[genre_pref["genre"] != "(no genres listed)"]
    user_genre_stats = (
        genre_pref.groupby("genre")
        .agg(Movies_Watched=("movieId", "count"), Avg_Rating=("rating", "mean"))
        .reset_index()
        .sort_values(by=["Movies_Watched", "Avg_Rating"], ascending=False)
        .head(8)
    )
    user_genre_stats["Avg_Rating"] = user_genre_stats["Avg_Rating"].round(2)

    # ALS dot-product inference
    user_row = user_factors[user_factors["id"] == user_id]
    if user_row.empty:
        return history_df, user_genre_stats, pd.DataFrame()

    u_vec = np.array(user_row.iloc[0]["features"], dtype=np.float64)
    i_mat = np.vstack(item_factors["features"].values)
    scores = i_mat @ u_vec

    preds_df = pd.DataFrame({
        "movieId": item_factors["id"].astype(int),
        "Predicted_Rating": np.clip(np.round(scores, 2), 0.5, 5.0),
        "Raw_ALS_Score": np.round(scores, 3),
    })

    watched_ids = set(user_ratings["movieId"])
    recs_df = (
        preds_df[~preds_df["movieId"].isin(watched_ids)]
        .merge(movies_df, on="movieId", how="inner")
        .sort_values(by="Raw_ALS_Score", ascending=False)
        .head(top_n)[["movieId", "title", "genres", "Predicted_Rating", "Raw_ALS_Score"]]
        .reset_index(drop=True)
    )

    return history_df, user_genre_stats, recs_df