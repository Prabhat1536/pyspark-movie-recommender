from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
from src.config import MODEL_DIR


def get_user_history(user_id: int, ratings_df: DataFrame, movies_df: DataFrame, limit: int = 10) -> DataFrame:
    return (
        ratings_df.filter(F.col("userId") == user_id)
        .join(movies_df, on="movieId", how="inner")
        .select("movieId", "title", "genres", "rating")
        .orderBy(F.desc("rating"))
        .limit(limit)
    )


def recommend_for_user(
    spark: SparkSession,
    user_id: int,
    ratings_df: DataFrame,
    movies_df: DataFrame,
    top_n: int = 10
) -> DataFrame:
    model = ALSModel.load(MODEL_DIR)
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    raw_recs = model.recommendForUserSubset(user_df, top_n + 50)

    if raw_recs.rdd.isEmpty():
        return spark.createDataFrame([], "movieId INT, title STRING, genres STRING, predicted_rating FLOAT")

    exploded_recs = raw_recs.select(
        F.col("userId"),
        F.explode("recommendations").alias("rec")
    ).select(
        F.col("userId"),
        F.col("rec.movieId").alias("movieId"),
        F.round(F.col("rec.rating"), 2).alias("predicted_rating")
    )

    already_watched = ratings_df.filter(F.col("userId") == user_id).select("movieId")
    unseen_recs = exploded_recs.join(already_watched, on="movieId", how="left_anti")

    final_recs = (
        unseen_recs.join(movies_df, on="movieId", how="inner")
        .select("movieId", "title", "genres", "predicted_rating")
        .orderBy(F.desc("predicted_rating"))
        .limit(top_n)
    )

    return final_recs
