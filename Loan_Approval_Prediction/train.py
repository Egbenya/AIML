import mlflow
import numpy as np
from data import X_train, X_val, y_train, y_val
from sklearn.model_selection import ParameterGrid
from params import rf_param_grid, grad_param_grid, xgb_param_grid, ada_param_grid, bag_param_grid
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from utils import eval_metrics

# Loop through the hyperparameter combinations and log results in separate runs
for params in ParameterGrid(rf_param_grid):
    with mlflow.start_run():

        model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = eval_metrics(y_val, y_pred)

        # print(f"metrics: {metrics}")

        # Logging the inputs such as dataset
        mlflow.log_input(
            mlflow.data.from_numpy(X_train.to_numpy()),
            context='Training dataset'
        )

        mlflow.log_input(
            mlflow.data.from_numpy(X_val.to_numpy()),
            context='Validation dataset'
        )

        # Logging hyperparameters
        mlflow.log_params(params)

        # Logging metrics
        mlflow.log_metrics(metrics)

        # Log the trained model
        mlflow.sklearn.log_model(
            model,
            "RandomForestClassifier",
            input_example=X_train,
            code_paths=['train.py','data.py','params.py','utils.py']
        )


