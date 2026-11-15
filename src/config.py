import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "als_movielens_model")

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_FOLDER = "ml-latest-small"

SPARK_APP_NAME = "MovieLens_ALS_Recommender"
DRIVER_MEMORY = "4g"

PARAM_GRID = {
    "rank": [10, 20],
    "regParam": [0.05, 0.1],
    "maxIter": [10, 15],
}
NUM_FOLDS = 3
SEED = 42
