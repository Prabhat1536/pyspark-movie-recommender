import shutil
import os
from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from src.config import MODEL_DIR, PARAM_GRID, NUM_FOLDS, SEED


def train_and_evaluate(ratings_df: DataFrame) -> ALSModel:
    train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=SEED)
    train_df.cache()
    test_df.cache()

    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
        seed=SEED
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank, PARAM_GRID["rank"])
        .addGrid(als.regParam, PARAM_GRID["regParam"])
        .addGrid(als.maxIter, PARAM_GRID["maxIter"])
        .build()
    )

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    cross_val = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=NUM_FOLDS,
        parallelism=2,
        seed=SEED
    )

    print(f"Training ALS model with {NUM_FOLDS}-fold cross-validation across {len(param_grid)} parameter combinations...")
    cv_model = cross_val.fit(train_df)
    best_model: ALSModel = cv_model.bestModel

    print("\n--- Best Model Hyperparameters ---")
    print(f"Rank (Latent Factors): {best_model.rank}")
    print(f"RegParam             : {best_model._java_obj.parent().getRegParam()}")
    print(f"MaxIter              : {best_model._java_obj.parent().getMaxIter()}")

    predictions = best_model.transform(test_df)
    rmse = evaluator.evaluate(predictions)
    print(f"Test Set RMSE        : {rmse:.4f}\n")

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    best_model.save(MODEL_DIR)
    print(f"Model saved successfully to: {MODEL_DIR}")

    train_df.unpersist()
    test_df.unpersist()

    return best_model
