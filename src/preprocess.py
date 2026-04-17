import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if "type" in df.columns:
        df["type"] = df["type"].astype("category").cat.codes

    df = df.dropna()

    return df


def split_features_target(df: pd.DataFrame, target_column: str = "label"):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y