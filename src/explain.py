"""Feature importance e SHAP (global + local)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import FEATURE_LABELS_PT, FIGURES_DIR, ensure_directories, slugify


def _label(name: str) -> str:
    return FEATURE_LABELS_PT.get(name, name.replace("_", " "))


def _transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    X_t = pipeline.named_steps["preprocess"].transform(X)
    return pd.DataFrame(X_t, columns=list(X.columns), index=getattr(X, "index", None))


def feature_importance_table(pipeline, feature_names: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impureza (Gini)"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
        kind = "|coeficiente| (após scaling)"
    else:
        raise ValueError(f"{type(model).__name__} não expõe importância nativa.")

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "label": [_label(f) for f in feature_names],
            "importance": values,
            "tipo": kind,
        }
    ).sort_values("importance", ascending=False)
    table["importance_norm"] = table["importance"] / table["importance"].sum()
    return table.reset_index(drop=True)


def plot_feature_importance(
    table: pd.DataFrame,
    model_name: str,
    top_n: int = 12,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    ensure_directories()
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["label"], top["importance"])
    ax.set_xlabel(table["tipo"].iloc[0])
    ax.set_title(f"Importância das features — {model_name}")
    fig.tight_layout()
    path = output_dir / f"feature_importance_{slugify(model_name)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def shap_explainer(pipeline, X_background: pd.DataFrame):
    """Cria explainer SHAP no espaço transformado (mesmo do modelo)."""
    model = pipeline.named_steps["model"]
    X_bg = _transformed_frame(pipeline, X_background)
    if hasattr(model, "estimators_"):
        return shap.TreeExplainer(model), X_bg
    return shap.Explainer(model, X_bg), X_bg


def shap_values_malignant(pipeline, X: pd.DataFrame, explainer=None, X_background=None):
    """Retorna Explanation SHAP da classe maligna."""
    if explainer is None:
        if X_background is None:
            X_background = X
        explainer, _ = shap_explainer(pipeline, X_background)

    X_t = _transformed_frame(pipeline, X)
    explanation = explainer(X_t)
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        # (n_samples, n_features, n_classes) — classe 1 = maligno
        malignant_values = values[:, :, 1]
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
        elif base_values.ndim == 1 and len(base_values) == 2:
            base_values = np.full(len(X_t), base_values[1])
    else:
        malignant_values = values
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim == 0:
            base_values = np.full(len(X_t), float(base_values))

    return shap.Explanation(
        values=malignant_values,
        base_values=base_values,
        data=np.asarray(X_t),
        feature_names=list(X.columns),
    )


def plot_shap_summary(
    explanation: shap.Explanation,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
    max_display: int = 12,
) -> Path:
    ensure_directories()
    labeled = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=[_label(n) for n in explanation.feature_names],
    )
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(labeled, max_display=max_display, show=False)
    plt.title(f"SHAP global — {model_name} (classe maligno)")
    plt.tight_layout()
    path = output_dir / f"shap_summary_{slugify(model_name)}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_waterfall(
    explanation: shap.Explanation,
    index: int,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
    suffix: str = "",
) -> Path:
    ensure_directories()
    row = explanation[index]
    labeled = shap.Explanation(
        values=row.values,
        base_values=row.base_values,
        data=row.data,
        feature_names=[_label(n) for n in explanation.feature_names],
    )
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(labeled, show=False)
    plt.title(f"SHAP local — {model_name}{suffix}")
    plt.tight_layout()
    tag = suffix.strip().replace(" ", "_").replace("—", "").strip("_") or str(index)
    path = output_dir / f"shap_waterfall_{slugify(model_name)}_{slugify(tag)}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def top_features_from_importance(table: pd.DataFrame, n: int = 10) -> list[str]:
    return table["feature"].head(n).tolist()
