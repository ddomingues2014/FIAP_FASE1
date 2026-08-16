"""Métricas e figuras de avaliação dos classificadores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import FIGURES_DIR, POSITIVE_LABEL, ensure_directories, slugify

sns.set_theme(style="whitegrid", context="talk")


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula métricas com foco na classe maligna (label=1)."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_maligno": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL)
        ),
        "precision_maligno": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL)
        ),
        "f1_maligno": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    report = classification_report(
        y_true,
        y_pred,
        target_names=["benigno", "maligno"],
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba)


def metrics_table(all_metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in all_metrics.items():
        rows.append(
            {
                "modelo": name,
                "accuracy": m["accuracy"],
                "recall_maligno": m["recall_maligno"],
                "precision_maligno": m["precision_maligno"],
                "f1_maligno": m["f1_maligno"],
                "roc_auc": m.get("roc_auc"),
            }
        )
    table = pd.DataFrame(rows).set_index("modelo")
    return table.round(4)


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    ensure_directories()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigno", "Maligno"],
        yticklabels=["Benigno", "Maligno"],
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {model_name}")
    fig.tight_layout()
    path = output_dir / f"confusion_{slugify(model_name)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_roc_curves(
    results: dict[str, tuple],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """results: nome -> (y_true, y_proba)."""
    ensure_directories()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Aleatório")
    ax.set_xlabel("Falso positivo (1 - especificidade)")
    ax.set_ylabel("Recall (sensibilidade)")
    ax.set_title("Curvas ROC — detecção de tumor maligno")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    path = output_dir / "roc_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_metrics_comparison(
    table: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    ensure_directories()
    plot_df = table[["accuracy", "recall_maligno", "precision_maligno", "f1_maligno"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Comparação de métricas no conjunto de teste")
    ax.legend(
        ["Accuracy", "Recall maligno", "Precision maligno", "F1 maligno"],
        loc="lower right",
        fontsize=9,
    )
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_metrics_json(all_metrics: dict, path: Path) -> None:
    serializable = {}
    for name, metrics in all_metrics.items():
        serializable[name] = {
            k: v
            for k, v in metrics.items()
            if k != "classification_report"
        }
        serializable[name]["classification_report"] = metrics["classification_report"]
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


