import os

os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = r'C:\hadoop\bin;' + os.environ.get('PATH', '')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'als_movielens_model')

DATASET_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
DATASET_FOLDER = 'ml-latest-small'

SPARK_APP_NAME = 'MovieLens_ALS_Recommender'
DRIVER_MEMORY = '4g'

PARAM_GRID = {
    'rank': [20],
    'regParam': [0.1],
    'maxIter': [15],
}
NUM_FOLDS = 2
SEED = 42
