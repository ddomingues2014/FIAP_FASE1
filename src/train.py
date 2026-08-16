"""Treino comparativo: regressão logística, Random Forest e KNN."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from .config import (
    ARTIFACTS_PATH,
    CV_FOLDS,
    FIGURES_DIR,
    METRICS_PATH,
    MODELS_DIR,
    POSITIVE_LABEL,
    RANDOM_STATE,
    ensure_directories,
)
from .data import dataset_overview, load_wisconsin, stratified_split
from .evaluate import (
    evaluate_model,
    metrics_table,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
    save_metrics_json,
)
from .explain import (
    feature_importance_table,
    plot_feature_importance,
    plot_shap_summary,
    plot_shap_waterfall,
    shap_values_malignant,
)
from .preprocess import build_model_pipeline

MODEL_SPECS = {
    "Regressão logística": {
        "estimator": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "param_grid": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [4, 8, None],
            "model__min_samples_split": [2, 5],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
        },
    },
}


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def train_one(name: str, X_train, y_train) -> tuple[GridSearchCV, dict]:
    spec = MODEL_SPECS[name]
    pipeline = build_model_pipeline(spec["estimator"])
    search = GridSearchCV(
        pipeline,
        param_grid=spec["param_grid"],
        scoring="recall",
        cv=_cv(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    info = {
        "best_params": search.best_params_,
        "best_cv_recall": float(search.best_score_),
    }
    return search, info


def select_best_model(all_metrics: dict[str, dict]) -> str:
    """Prioriza recall do maligno; desempate por F1 e depois ROC-AUC."""
    ranked = sorted(
        all_metrics.items(),
        key=lambda item: (
            item[1]["recall_maligno"],
            item[1]["f1_maligno"],
            item[1].get("roc_auc", 0.0),
        ),
        reverse=True,
    )
    return ranked[0][0]


def train_and_evaluate(
    persist: bool = True,
    output_dir: Path = FIGURES_DIR,
) -> dict:
    ensure_directories()
    X, y = load_wisconsin()
    overview = dataset_overview(X, y)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    models = {}
    search_info = {}
    all_metrics = {}
    roc_payload = {}

    for name in MODEL_SPECS:
        search, info = train_one(name, X_train, y_train)
        pipeline = search.best_estimator_
        models[name] = pipeline
        search_info[name] = info
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["best_params"] = info["best_params"]
        metrics["best_cv_recall"] = info["best_cv_recall"]
        all_metrics[name] = metrics
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_payload[name] = (y_test, y_proba)
        plot_confusion_matrix(y_test, y_pred, name, output_dir=output_dir)

    table = metrics_table(all_metrics)
    plot_metrics_comparison(table, output_dir=output_dir)
    plot_roc_curves(roc_payload, output_dir=output_dir)

    best_name = select_best_model(all_metrics)
    importance_tables = {}
    for name in ("Regressão logística", "Random Forest"):
        imp = feature_importance_table(models[name], list(X.columns))
        importance_tables[name] = imp
        plot_feature_importance(imp, name, output_dir=output_dir)

    # SHAP no melhor modelo interpretável com árvore; se o melhor for KNN, usa RF.
    shap_model_name = best_name if best_name != "KNN" else "Random Forest"
    shap_pipeline = models[shap_model_name]
    background = X_train.sample(n=min(80, len(X_train)), random_state=RANDOM_STATE)
    shap_test = X_test.copy()
    explanation = shap_values_malignant(
        shap_pipeline,
        shap_test,
        X_background=background,
    )
    plot_shap_summary(explanation, shap_model_name, output_dir=output_dir)

    y_test_arr = y_test.to_numpy()
    pred_test = shap_pipeline.predict(X_test)
    malignant_hits = [
        i for i, (yt, yp) in enumerate(zip(y_test_arr, pred_test)) if yt == POSITIVE_LABEL and yp == POSITIVE_LABEL
    ]
    benign_hits = [
        i for i, (yt, yp) in enumerate(zip(y_test_arr, pred_test)) if yt == 0 and yp == 0
    ]
    local_indices = {}
    if malignant_hits:
        idx = malignant_hits[0]
        plot_shap_waterfall(
            explanation,
            idx,
            shap_model_name,
            output_dir=output_dir,
            suffix="caso maligno",
        )
        local_indices["maligno"] = int(X_test.index[idx])
    if benign_hits:
        idx = benign_hits[0]
        plot_shap_waterfall(
            explanation,
            idx,
            shap_model_name,
            output_dir=output_dir,
            suffix="caso benigno",
        )
        local_indices["benigno"] = int(X_test.index[idx])

    artifacts = {
        "models": models,
        "metrics": all_metrics,
        "metrics_table": table,
        "best_model_name": best_name,
        "shap_model_name": shap_model_name,
        "feature_names": list(X.columns),
        "overview": overview,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "importance_tables": importance_tables,
        "search_info": search_info,
        "shap_local_indices": local_indices,
        "positive_label": POSITIVE_LABEL,
    }

    if persist:
        joblib.dump(artifacts, ARTIFACTS_PATH)
        save_metrics_json(all_metrics, METRICS_PATH)
        table.to_csv(MODELS_DIR / "metrics_table.csv")
        summary = {
            "best_model_name": best_name,
            "shap_model_name": shap_model_name,
            "overview": overview,
            "search_info": search_info,
            "shap_local_indices": local_indices,
        }
        (MODELS_DIR / "training_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    return artifacts


def load_artifacts(path: Path = ARTIFACTS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Artefatos não encontrados em {path}. Rode: python -m src.train"
        )
    return joblib.load(path)


def main() -> None:
    artifacts = train_and_evaluate(persist=True)
    print("=== Resumo do dataset ===")
    print(json.dumps(artifacts["overview"], indent=2, ensure_ascii=False))
    print("\n=== Métricas no teste ===")
    print(artifacts["metrics_table"].to_string())
    print(f"\nMelhor modelo (recall maligno): {artifacts['best_model_name']}")
    print(f"Modelo usado no SHAP: {artifacts['shap_model_name']}")
    print(f"Artefatos: {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
