import mlflow

experiment_name = "RandomForestClassifier"
entry_point = "Training"

mlflow.set_tracking_uri("http://127.0.0.1:5000")#"http://ec2-3-91-59-37.compute-1.amazonaws.com:5000/") #

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name,
    env_manager="conda"
)
