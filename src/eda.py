"""Gráficos de exploração do Wisconsin Diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FEATURE_LABELS_PT, FIGURES_DIR, KEY_FEATURES, ensure_directories

sns.set_theme(style="whitegrid", context="talk")


def _label(name: str) -> str:
    return FEATURE_LABELS_PT.get(name, name.replace("_", " "))


def _with_diagnosis(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    frame = X.copy()
    frame["diagnostico"] = y.map({0: "Benigno", 1: "Maligno"})
    return frame


def plot_class_balance(y: pd.Series, output_dir: Path = FIGURES_DIR) -> Path:
    ensure_directories()
    counts = y.map({0: "Benigno", 1: "Maligno"}).value_counts().reindex(
        ["Benigno", "Maligno"]
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#4C9F38", "#C0392B"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Pacientes")
    ax.set_title("Distribuição das classes (diagnóstico)")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{int(value)}\n({value / counts.sum():.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    path = output_dir / "eda_class_balance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_key_feature_boxplots(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    ensure_directories()
    features = features or KEY_FEATURES[:6]
    frame = _with_diagnosis(X[features], y)
    melted = frame.melt(id_vars="diagnostico", var_name="feature", value_name="valor")
    melted["feature"] = melted["feature"].map(_label)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=melted,
        x="feature",
        y="valor",
        hue="diagnostico",
        palette={"Benigno": "#4C9F38", "Maligno": "#C0392B"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Valor")
    ax.set_title("Features clínicas por diagnóstico")
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "eda_boxplots_key_features.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_key_feature_histograms(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str] | None = None,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    ensure_directories()
    features = features or [
        "mean_radius",
        "mean_concave_points",
        "worst_area",
        "worst_concave_points",
    ]
    frame = _with_diagnosis(X[features], y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feature in zip(axes.ravel(), features):
        sns.histplot(
            data=frame,
            x=feature,
            hue="diagnostico",
            bins=30,
            kde=True,
            palette={"Benigno": "#4C9F38", "Maligno": "#C0392B"},
            ax=ax,
            stat="density",
            common_norm=False,
        )
        ax.set_xlabel(_label(feature))
        ax.set_title(_label(feature))
    fig.suptitle("Distribuições condicionais ao diagnóstico", y=1.01)
    fig.tight_layout()
    path = output_dir / "eda_histograms_key_features.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_heatmap(
    X: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    top_features: list[str] | None = None,
) -> Path:
    """Heatmap completo fica ilegível; usamos um recorte clínico + versão full reduzida."""
    ensure_directories()
    cols = top_features or KEY_FEATURES
    corr = X[cols].corr()
    labels = [_label(c) for c in cols]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Correlação entre features clínicas (recortes mean/worst)")
    fig.tight_layout()
    path = output_dir / "eda_correlation_key.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        X.corr(),
        cmap="vlag",
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={"label": "correlação de Pearson"},
    )
    ax.set_title("Matriz de correlação completa (30 features FNA)")
    fig.tight_layout()
    full_path = output_dir / "eda_correlation_full.png"
    fig.savefig(full_path, dpi=160)
    plt.close(fig)
    return path


def plot_mean_vs_worst_scatter(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    ensure_directories()
    frame = _with_diagnosis(X, y)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=frame,
        x="mean_radius",
        y="worst_radius",
        hue="diagnostico",
        palette={"Benigno": "#4C9F38", "Maligno": "#C0392B"},
        ax=ax,
        alpha=0.75,
    )
    ax.set_xlabel(_label("mean_radius"))
    ax.set_ylabel(_label("worst_radius"))
    ax.set_title("Colinearidade: raio médio vs. pior raio")
    ax.legend(title="")
    fig.tight_layout()
    path = output_dir / "eda_scatter_radius.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def export_eda_figures(X: pd.DataFrame, y: pd.Series, output_dir: Path = FIGURES_DIR) -> list[Path]:
    return [
        plot_class_balance(y, output_dir),
        plot_key_feature_boxplots(X, y, output_dir=output_dir),
        plot_key_feature_histograms(X, y, output_dir=output_dir),
        plot_correlation_heatmap(X, output_dir=output_dir),
        plot_mean_vs_worst_scatter(X, y, output_dir=output_dir),
    ]


def main() -> None:
    from .data import dataset_overview, load_wisconsin

    X, y = load_wisconsin()
    overview = dataset_overview(X, y)
    print("=== Wisconsin Diagnostic ===")
    print(f"Amostras: {overview['n_samples']} | Features: {overview['n_features']}")
    print(f"Malignos: {overview['n_malignant']} | Benignos: {overview['n_benign']}")
    print(f"Missing: {overview['missing_values']}")
    print("\nEstatísticas descritivas (features clínicas):")
    print(X[KEY_FEATURES].describe().T.round(3).to_string())
    paths = export_eda_figures(X, y)
    print("\nFiguras geradas:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
