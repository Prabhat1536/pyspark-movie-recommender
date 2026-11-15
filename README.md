# PySpark Movie Recommendation Engine

A scalable collaborative filtering recommendation engine built with **PySpark MLlib (ALS)** on the MovieLens dataset.

## Features
- Automated MovieLens dataset download & schema validation
- Alternating Least Squares (ALS) matrix factorization with K-Fold Cross-Validation (RMSE)
- Unseen-item filtering via Spark SQL left_anti joins
- Saved ALSModel persistence for low-latency inference

## Quickstart
`ash
python -m pip install -r requirements.txt
python main.py --train --user-id 1 --top-n 10
`
