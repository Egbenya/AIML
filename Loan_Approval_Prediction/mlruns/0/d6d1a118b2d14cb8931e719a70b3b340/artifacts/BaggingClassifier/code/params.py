import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Hyperparameter tuning
rf_param_grid = {
    'n_estimators': np.arange(50, 110, 25),
    'criterion': ['gini', 'entropy'],
    'min_samples_split': np.arange(1,5),
    'min_samples_leaf': np.arange(1,5),
    'max_samples': np.arange(0.4, 0.7, 0.1),
    'max_leaf_nodes': [4,10,20,50,None],
}

grad_param_grid = {
    'n_estimators': np.arange(50, 110, 25),
    'learning_rate': [0.01, 0.1, 0.2, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_features': [0.5, 0.7, 1],
}

xgb_param_grid = {
    'n_estimators': [100],#np.arange(50, 110, 25),
    'scale_pos_weight': [1, 2, 5],
    'learning_rate': [0.01, 0.1, 0.05],
    'gamma': [1, 3],
    'subsample': [0.7, 0.9],
}

ada_param_grid = {
    'n_estimators': np.arange(50, 110, 25),
    'learning_rate': [0.01, 0.1, 0.05],
    'estimator': [
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}

bag_param_grid = {
    'max_samples': [0.8]#[0.8, 0.9, 1],
    # 'max_features': [0.5, 0.7, 1],
    # 'n_estimators': [50, 70, 100],
}