import os
import zipfile
import requests
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType
from src.config import DATA_DIR, DATASET_URL, DATASET_FOLDER, SPARK_APP_NAME, DRIVER_MEMORY


def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def download_and_extract_data() -> str:
    target_path = os.path.join(DATA_DIR, DATASET_FOLDER)
    if os.path.exists(target_path):
        return target_path

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"{DATASET_FOLDER}.zip")

    print(f"Downloading MovieLens dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    os.remove(zip_path)
    return target_path


def load_datasets(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    dataset_path = download_and_extract_data()

    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True),
    ])

    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
    ])

    ratings_df = spark.read.csv(
        os.path.join(dataset_path, "ratings.csv"),
        header=True,
        schema=ratings_schema
    ).na.drop()

    movies_df = spark.read.csv(
        os.path.join(dataset_path, "movies.csv"),
        header=True,
        schema=movies_schema
    ).na.drop()

    return ratings_df, movies_df


def print_dataset_stats(ratings_df: DataFrame) -> None:
    total_ratings = ratings_df.count()
    num_users = ratings_df.select("userId").distinct().count()
    num_movies = ratings_df.select("movieId").distinct().count()
    sparsity = (1.0 - (total_ratings / (num_users * num_movies))) * 100

    print("\n--- Dataset Statistics ---")
    print(f"Total Ratings  : {total_ratings:,}")
    print(f"Unique Users   : {num_users:,}")
    print(f"Unique Movies  : {num_movies:,}")
    print(f"Matrix Sparsity: {sparsity:.2f}%\n")
