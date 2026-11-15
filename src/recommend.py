import os
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from src.config import MODEL_DIR


def get_user_history(user_id: int, ratings_df: DataFrame, movies_df: DataFrame, limit: int = 10) -> DataFrame:
    return (
        ratings_df.filter(F.col("userId") == user_id)
        .join(movies_df, on="movieId", how="inner")
        .select("movieId", "title", "genres", "rating")
        .orderBy(F.desc("rating"), F.asc("title"))
        .limit(limit)
    )


def recommend_for_user(
    spark: SparkSession,
    user_id: int,
    ratings_df: DataFrame,
    movies_df: DataFrame,
    top_n: int = 10
) -> DataFrame:
    user_path = os.path.join(MODEL_DIR, "user_factors.parquet")
    item_path = os.path.join(MODEL_DIR, "item_factors.parquet")
    if not os.path.exists(user_path) or not os.path.exists(item_path):
        raise FileNotFoundError(f"No saved model found at {MODEL_DIR}. Run with --train first.")

    user_pdf = pd.read_parquet(user_path)
    item_pdf = pd.read_parquet(item_path)

    user_row = user_pdf[user_pdf["id"] == user_id]
    if user_row.empty:
        return spark.createDataFrame([], schema="movieId INT, title STRING, genres STRING, predicted_rating DOUBLE")

    u_vec = np.array(user_row.iloc[0]["features"], dtype=np.float64)
    i_mat = np.vstack(item_pdf["features"].values)
    scores = i_mat @ u_vec

    preds_pdf = pd.DataFrame({
        "movieId": item_pdf["id"].astype(int),
        "predicted_rating": np.round(scores, 2)
    })

    raw_recs_df = spark.createDataFrame(preds_pdf)
    watched_df = ratings_df.filter(F.col("userId") == user_id).select("movieId")

    unseen_recs_df = (
        raw_recs_df.join(watched_df, on="movieId", how="left_anti")
        .join(movies_df, on="movieId", how="inner")
        .select("movieId", "title", "genres", "predicted_rating")
        .orderBy(F.desc("predicted_rating"))
        .limit(top_n)
    )
    return unseen_recs_df