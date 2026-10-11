from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def train_random_forest(X_train, y_train):
    random_forest_params = {
        "n_estimators": [200, 300],
        "max_depth": [5, 8, None],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 3, 5],
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced"),
        param_distributions=random_forest_params,
        n_iter=20,
        cv=5,
        n_jobs=-1,
        scoring="f1",
        random_state=49,
    )

    rf_search.fit(X_train, y_train)
    return rf_search.best_estimator_, rf_search.best_params_, rf_search.best_score_


def train_xgboost(X_train, y_train):
    xgboost_params = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.1, 0.3, 0.5, 0.7],
        "max_depth": [4, 6, 8, 10],
        "subsample": [0.5, 1],
        "colsample_bytree": [0.5, 0.75, 1],
        "gamma": [0, 0.5, 1],
        "reg_lambda": [1, 2, 3],
    }
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(use_label_encoder=False),
        param_distributions=xgboost_params,
        n_iter=20,
        cv=3,
        n_jobs=-1,
        scoring="f1",
        random_state=49,
    )

    xgb_search.fit(X_train, y_train)
    return xgb_search.best_estimator_, xgb_search.best_params_, xgb_search.best_score_
