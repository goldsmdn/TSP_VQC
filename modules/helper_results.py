import pandas as pd
from pathlib import Path

from modules.config import (
    RESULTS_DIR,
    RESULTS_FILE,
)


def apply_special_processing(df: pd.DataFrame) -> pd.DataFrame:
    df['noise'] = df['noise'].fillna(False)
    return df


def read_data():
    """Read data from csv file into pandas dataframe"""
    results_path = Path(RESULTS_DIR).joinpath(RESULTS_FILE)
    df = pd.read_csv(results_path)
    df = apply_special_processing(df)
    return df


def find_quality(
    df: pd.DataFrame, factor: float = 1, round: bool = None
) -> pd.DataFrame:
    """Find the quantum and error metrics"""
    df['quality'] = factor * df['best_dist'] / df['best_dist_found']
    df['error'] = 1 * factor - df['quality']
    if round:
        df['quality'] = df['quality'].round(round)
        df['error'] = df['error'].round(round)
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply flexible filters"""
    for col, val in filters.items():
        if callable(val):
            df = df[val(df[col])]
        elif isinstance(val, list):
            df = df[df[col].isin(val)]
        else:
            df = df[df[col] == val]
    return df
