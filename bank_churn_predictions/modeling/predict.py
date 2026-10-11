import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def score_function(model, X_test, y_test):
    predicted_values = model.predict(X_test)
    report_df = pd.DataFrame(
        columns=[
            "Accuracy Score",
            "Precision",
            "Recall",
            "F1 Score",
            "Specificity",
            "Balanced Accuracy Score",
        ]
    )
    report_df.loc[0] = (
        accuracy_score(y_test, predicted_values),
        precision_score(y_test, predicted_values),
        recall_score(y_test, predicted_values),
        f1_score(y_test, predicted_values),
        recall_score(y_test, predicted_values, pos_label=0),
        balanced_accuracy_score(y_test, predicted_values),
    )
    return report_df


def predict_using_model(model, X_test, y_test):
    logger.info("Making predictions")

    return model.predict(X_test)
