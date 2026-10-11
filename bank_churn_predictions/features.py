import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X = X.drop(
        columns=[
            "Exited",
            "RowNumber",
            "CustomerId",
            "Surname",
        ],
        errors="ignore",
    ).copy()

    X["HasBalance"] = (X["Balance"] > 0).astype(int)

    X["Age_Group"] = pd.cut(
        X["Age"],
        bins=np.linspace(df.Age.min(), df.Age.max(), 7),
        labels=[x for x in range(1, 7)],
        include_lowest=True,
    )

    X["Active_x_Balance"] = X["IsActiveMember"] * X["HasBalance"]

    return X


def create_target(df: pd.DataFrame) -> pd.Series:
    return df["Exited"].copy()
