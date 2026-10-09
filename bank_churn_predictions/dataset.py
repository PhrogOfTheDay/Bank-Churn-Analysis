from pathlib import Path

from loguru import logger
import pandas as pd

from bank_churn_predictions.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_stage1_data(
    input_path: Path = INTERIM_DATA_DIR / "stage_1_preprocessing.csv",
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info("Loading stage 1 dataset from {}", input_path)

    df = pd.read_csv(input_path)

    logger.info(
        "Loaded {} rows and {} columns",
        len(df),
        len(df.columns),
    )

    return df


def load_raw_data(
    input_path: Path = RAW_DATA_DIR / "Churn_Modelling.csv",
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info("Loading raw dataset from {}", input_path)

    df = pd.read_csv(input_path)

    logger.info(
        "Loaded {} rows and {} columns",
        len(df),
        len(df.columns),
    )

    return df


def load_train_data(
    input_path: Path = PROCESSED_DATA_DIR / "train_data.csv",
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info("Loading train dataset from {}", input_path)

    df = pd.read_csv(input_path)

    logger.info(
        "Loaded {} rows and {} columns",
        len(df),
        len(df.columns),
    )

    return df


def load_test_data(
    input_path: Path = PROCESSED_DATA_DIR / "test_data.csv",
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info("Loading test dataset from {}", input_path)

    df = pd.read_csv(input_path)

    logger.info(
        "Loaded {} rows and {} columns",
        len(df),
        len(df.columns),
    )

    return df


def load_smote_data(
    input_path: Path = PROCESSED_DATA_DIR / "SMOTE.csv",
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info("Loading SMOTE dataset from {}", input_path)

    df = pd.read_csv(input_path)

    logger.info(
        "Loaded {} rows and {} columns",
        len(df),
        len(df.columns),
    )

    return df
