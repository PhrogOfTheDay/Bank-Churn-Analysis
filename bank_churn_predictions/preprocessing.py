# preprocessing.py

import re

from imblearn.over_sampling import SMOTENC
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "minmax",
                MinMaxScaler(),
                ["Tenure", "NumOfProducts"],
            ),
            (
                "standard",
                StandardScaler(),
                ["CreditScore", "EstimatedSalary"],
            ),
            (
                "robust",
                RobustScaler(),
                ["Age", "Balance"],
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop_first=True),
                ["Geography"],
            ),
        ],
        remainder="passthrough",
    )


def preprocess_data(preprocessor, X_train, y_train, X_test, y_test):
    preprocessor.fit(X_train)
    transformed_feature_names = preprocessor.get_feature_names_out()
    for index, val in enumerate(transformed_feature_names):
        transformed_feature_names[index] = re.sub(
            r"^([a-z]*__)", "", transformed_feature_names[index]
        )

    preprocessed_X_train = pd.DataFrame(
        preprocessor.transform(X_train), columns=transformed_feature_names
    )
    preprocessed_X_test = pd.DataFrame(
        preprocessor.transform(X_test), columns=transformed_feature_names
    )
    return preprocessed_X_train, preprocessed_X_test


def create_smote(preprocessed_X_train, y_train):
    cat_indices = [
        preprocessed_X_train.columns.get_loc(c)
        for c in [
            "HasBalance",
            "Age_Group",
            "Active_x_Balance",
            "Germany",
            "Spain",
            "Gender",
            "HasCrCard",
            "IsActiveMember",
        ]
    ]

    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_resampled, y_resampled = smotenc.fit_resample(preprocessed_X_train, y_train)
    return X_resampled, y_resampled
