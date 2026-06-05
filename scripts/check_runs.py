import mlflow

mlflow.set_tracking_uri(
    "sqlite:///outputs/mlflow/mlflow.db"
)

exp = mlflow.get_experiment_by_name(
    "BDD100K_YOLO"
)

runs = mlflow.search_runs(
    [exp.experiment_id]
)

print(runs.columns.tolist())