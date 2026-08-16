"""Carga do Breast Cancer Wisconsin e split estratificado."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


def _to_snake_case(name: str) -> str:
    return name.strip().replace(" ", "_")


def load_wisconsin() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o dataset UCI via scikit-learn.

    O sklearn usa 0 = maligno e 1 = benigno. Remapeamos para:
    - 1 = maligno (classe positiva, o que queremos detectar)
    - 0 = benigno

    Assim, recall/F1 da classe 1 correspondem ao risco de câncer.
    """
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    feature_cols = [_to_snake_case(name) for name in bunch.feature_names]
    X = frame.drop(columns=["target"])
    X.columns = feature_cols

    # sklearn: target 0 = malignant, 1 = benign
    y = (bunch.target.to_numpy() == 0).astype(int)
    y = pd.Series(y, index=X.index, name="diagnosis")
    return X, y


def dataset_overview(X: pd.DataFrame, y: pd.Series) -> dict:
    """Resumo usado no relatório, no app e nos scripts de EDA."""
    n_malignant = int((y == 1).sum())
    n_benign = int((y == 0).sum())
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_malignant": n_malignant,
        "n_benign": n_benign,
        "malignant_rate": float(n_malignant / len(y)),
        "missing_values": int(X.isna().sum().sum()),
        "feature_names": list(X.columns),
        "source": "UCI Breast Cancer Wisconsin Diagnostic (sklearn)",
        "kaggle_equivalent": (
            "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data"
        ),
    }


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino/teste preservando a proporção maligno/benigno."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
