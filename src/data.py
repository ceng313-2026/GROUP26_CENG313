"""Data loading and preparation helpers for the CENG313 project."""

from pathlib import Path
import pandas as pd


def load_competition_data(data_dir):
    """Load train, test, and sample-submission files from data_dir."""
    data_dir = Path(data_dir)
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def prepare_modeling_data(train):
    """Remove empty prompts and duplicate prompt-response pairs.

    The original DataFrame is not modified.
    """
    required = {"prompt", "response_a", "response_b", "winner"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    modeling_data = train.copy()
    text_columns = ["prompt", "response_a", "response_b"]
    for column in text_columns:
        modeling_data[column] = modeling_data[column].fillna("").astype(str)

    modeling_data = modeling_data[
        modeling_data["prompt"].str.strip().ne("")
    ].copy()

    modeling_data = modeling_data.drop_duplicates(
        subset=["prompt", "response_a", "response_b"],
        keep="first",
    ).reset_index(drop=True)

    return modeling_data
