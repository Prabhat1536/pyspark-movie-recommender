import argparse
import os
from src.config import MODEL_DIR
from src.data_loader import get_spark_session, load_datasets, print_dataset_stats
from src.train import train_and_evaluate
from src.recommend import get_user_history, recommend_for_user


def main():
    parser = argparse.ArgumentParser(description="PySpark Collaborative Filtering Movie Recommender")
    parser.add_argument("--train", action="store_true", help="Force re-training of the ALS model")
    parser.add_argument("--user-id", type=int, default=1, help="Target User ID for recommendations")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to display")
    args = parser.parse_args()

    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        ratings_df, movies_df = load_datasets(spark)
        print_dataset_stats(ratings_df)

        if args.train or not os.path.exists(MODEL_DIR):
            train_and_evaluate(ratings_df)
        else:
            print(f"Using existing trained model at: {MODEL_DIR}")

        print(f"\n=== User {args.user_id}: Top Historical Ratings ===")
        history_df = get_user_history(args.user_id, ratings_df, movies_df, limit=5)
        history_df.show(truncate=False)

        print(f"=== User {args.user_id}: Top {args.top_n} Recommended Movies ===")
        recs_df = recommend_for_user(spark, args.user_id, ratings_df, movies_df, top_n=args.top_n)
        recs_df.show(truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
