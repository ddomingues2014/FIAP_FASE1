"""Exploração do Wisconsin Diagnostic — script de demonstração (substitui notebook)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, KEY_FEATURES
from src.data import dataset_overview, load_wisconsin, stratified_split
from src.eda import export_eda_figures
from src.preprocess import build_preprocess_pipeline


def main() -> None:
    X, y = load_wisconsin()
    overview = dataset_overview(X, y)

    print("=" * 72)
    print("EDA — Breast Cancer Wisconsin Diagnostic (UCI / Kaggle)")
    print("=" * 72)
    print(f"Fonte sklearn: {overview['source']}")
    print(f"Equivalente Kaggle: {overview['kaggle_equivalent']}")
    print(f"Amostras: {overview['n_samples']} | Features: {overview['n_features']}")
    print(f"Malignos (1): {overview['n_malignant']} ({overview['malignant_rate']:.1%})")
    print(f"Benignos (0): {overview['n_benign']}")
    print(f"Valores ausentes: {overview['missing_values']}")

    print("\n--- Estatísticas descritivas (features clínicas) ---")
    print(X[KEY_FEATURES].describe().T.round(3).to_string())

    print("\n--- Padrão clínico ---")
    print(
        "Tumores malignos tendem a maior raio, área, perímetro e concavidade. "
        "Isso é coerente com a literatura do FNA: núcleos maiores e mais "
        "irregulares associam-se a malignidade. Há sobreposição entre classes; "
        "por isso o modelo apoia, mas não substitui, o médico."
    )

    by_diag = X[KEY_FEATURES].groupby(y.map({0: "benigno", 1: "maligno"})).mean().round(3)
    print("\nMédias por diagnóstico:")
    print(by_diag.T.to_string())

    print("\n--- Correlação / colinearidade ---")
    corr = X[["mean_radius", "mean_perimeter", "mean_area", "worst_radius", "worst_area"]].corr()
    print(corr.round(3).to_string())
    print(
        "\nRaio, perímetro e área são quase colineares. Mantemos as 30 features: "
        "a logística usa L2 e o Random Forest tolera redundância. PCA fica como "
        "melhoria futura, não como requisito desta fase."
    )

    print("\n--- Pré-processamento e split estratificado ---")
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    pipe = build_preprocess_pipeline()
    X_train_s = pipe.fit_transform(X_train)
    pipe.transform(X_test)
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"Proporção maligno treino: {y_train.mean():.3f} | teste: {y_test.mean():.3f}")
    print(f"Média após scaler (5 primeiras): {X_train_s.mean(axis=0)[:5].round(4)}")
    print(f"Desvio após scaler (5 primeiras): {X_train_s.std(axis=0)[:5].round(4)}")
    print(
        "Pipeline: SimpleImputer(median) + StandardScaler. O imputador documenta "
        "o tratamento de missing (Wisconsin sklearn não tem NA) e o scaler é "
        "obrigatório para KNN / útil na logística. Tudo dentro do Pipeline para "
        "não vazar estatísticas do teste."
    )

    paths = export_eda_figures(X, y, FIGURES_DIR)
    print("\n--- Figuras em reports/figures ---")
    for path in paths:
        print(f"  {path.name}")
    print("\nPróximo passo: python -m src.train   ou   python scripts/export_figures.py")


if __name__ == "__main__":
    main()
