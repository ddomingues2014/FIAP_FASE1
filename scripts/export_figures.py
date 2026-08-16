"""Gera figuras de EDA, métricas e SHAP de forma reprodutível."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, ensure_directories
from src.data import load_wisconsin
from src.eda import export_eda_figures
from src.train import train_and_evaluate


def main() -> None:
    ensure_directories()
    X, y = load_wisconsin()
    eda_paths = export_eda_figures(X, y, FIGURES_DIR)
    print("Figuras de EDA:")
    for path in eda_paths:
        print(f"  - {path}")

    artifacts = train_and_evaluate(persist=True, output_dir=FIGURES_DIR)
    print("\nMétricas:")
    print(artifacts["metrics_table"].to_string())
    print(f"\nFiguras em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
