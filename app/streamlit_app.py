"""App de suporte ao diagnóstico — Tech Challenge IADT Fase 1."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DISCLAIMER, FEATURE_LABELS_PT, FIGURES_DIR, KEY_FEATURES, slugify
from src.explain import shap_values_malignant
from src.train import load_artifacts

st.set_page_config(
    page_title="Suporte ao diagnóstico — câncer de mama",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def get_artifacts() -> dict:
    return load_artifacts()


def _label(name: str) -> str:
    return FEATURE_LABELS_PT.get(name, name.replace("_", " "))


def _show_disclaimer() -> None:
    st.warning(DISCLAIMER)


def _preset_options(artifacts: dict) -> dict[str, pd.Series]:
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    options: dict[str, pd.Series] = {
        "Mediana do treino (referência)": artifacts["X_train"].median(),
    }
    mal = X_test[y_test == 1]
    ben = X_test[y_test == 0]
    if len(mal):
        options[f"Caso de teste maligno (índice {mal.index[0]})"] = mal.iloc[0]
    if len(ben):
        options[f"Caso de teste benigno (índice {ben.index[0]})"] = ben.iloc[0]
    if len(mal) > 1:
        options[f"Caso de teste maligno (índice {mal.index[1]})"] = mal.iloc[1]
    return options


def page_prediction(artifacts: dict) -> None:
    st.title("Predição de risco (maligno vs. benigno)")
    st.caption(
        "Classificador treinado no Breast Cancer Wisconsin Diagnostic (UCI). "
        "Preencha as medidas do FNA ou carregue um caso de teste."
    )
    _show_disclaimer()

    best_name = artifacts["best_model_name"]
    shap_name = artifacts["shap_model_name"]
    pipeline = artifacts["models"][best_name]
    shap_pipeline = artifacts["models"][shap_name]
    feature_names = artifacts["feature_names"]

    presets = _preset_options(artifacts)
    preset_name = st.selectbox("Caso base", list(presets.keys()))
    base_row = presets[preset_name]

    st.subheader("Ajustar features clínicas principais")
    cols = st.columns(2)
    edited = {}
    for i, feature in enumerate(KEY_FEATURES):
        with cols[i % 2]:
            default = float(base_row[feature])
            edited[feature] = st.number_input(
                _label(feature),
                value=default,
                format="%.4f",
                key=f"feat_{feature}",
            )

    with st.expander("Demais features (valores do caso base)", expanded=False):
        extra_cols = st.columns(3)
        extra_features = [f for f in feature_names if f not in KEY_FEATURES]
        for i, feature in enumerate(extra_features):
            with extra_cols[i % 3]:
                edited[feature] = st.number_input(
                    _label(feature),
                    value=float(base_row[feature]),
                    format="%.4f",
                    key=f"extra_{feature}",
                )

    row = pd.DataFrame([{f: edited[f] for f in feature_names}])

    if st.button("Calcular predição", type="primary"):
        proba = float(pipeline.predict_proba(row)[0, 1])
        pred = int(pipeline.predict(row)[0])
        label = "Maligno" if pred == 1 else "Benigno"

        left, right = st.columns([1, 1])
        with left:
            st.metric("Modelo usado", best_name)
            st.metric("Classe predita", label)
            st.metric("Probabilidade de maligno", f"{proba:.1%}")
            st.progress(min(max(proba, 0.0), 1.0))
            if pred == 1:
                st.error(
                    "Risco elevado de malignidade segundo o modelo. "
                    "Encaminhar para avaliação médica completa."
                )
            else:
                st.success(
                    "Padrão mais compatível com benigno segundo o modelo. "
                    "Isso não descarta acompanhamento clínico."
                )

        with right:
            st.markdown(f"**Explicação local (SHAP) — {shap_name}**")
            background = artifacts["X_train"].sample(
                n=min(80, len(artifacts["X_train"])),
                random_state=42,
            )
            explanation = shap_values_malignant(
                shap_pipeline,
                row,
                X_background=background,
            )
            labeled = shap.Explanation(
                values=explanation[0].values,
                base_values=explanation[0].base_values,
                data=explanation[0].data,
                feature_names=[_label(n) for n in explanation.feature_names],
            )
            fig = plt.figure(figsize=(8, 5))
            shap.plots.waterfall(labeled, show=False)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        st.info(
            "Barras à direita no waterfall aumentam a chance de **maligno**; "
            "à esquerda empurram para **benigno**. A soma parte do valor base do modelo."
        )


def page_analysis(artifacts: dict) -> None:
    st.title("Análise do modelo")
    st.caption("Exploração, métricas comparativas e explicabilidade global.")
    _show_disclaimer()

    overview = artifacts["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", overview["n_samples"])
    c2.metric("Features", overview["n_features"])
    c3.metric("Malignos", overview["n_malignant"])
    c4.metric("Benignos", overview["n_benign"])

    st.subheader("Por que essas métricas?")
    st.markdown(
        """
No diagnóstico de câncer, **falso negativo** (tumor maligno classificado como benigno)
é o erro mais grave: atrasa biópsia e tratamento. Por isso a métrica principal é o
**recall da classe maligna**, seguida de **F1** e da matriz de confusão.

A **accuracy** entra na comparação, mas sozinha é insuficiente: o conjunto tem
mais casos benignos (~63%) e o custo dos erros é assimétrico. ROC-AUC e PR-AUC
complementam a visão de ranking.
"""
    )

    st.subheader("Comparação dos modelos (conjunto de teste)")
    table = artifacts["metrics_table"].copy()
    table.columns = [
        "Accuracy",
        "Recall maligno",
        "Precision maligno",
        "F1 maligno",
        "ROC-AUC",
    ]
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
    st.success(f"Modelo selecionado para a predição: **{artifacts['best_model_name']}**")

    fig_cols = st.columns(2)
    images = [
        ("Balanceamento das classes", FIGURES_DIR / "eda_class_balance.png"),
        ("Boxplots das features clínicas", FIGURES_DIR / "eda_boxplots_key_features.png"),
        ("Distribuições condicionais", FIGURES_DIR / "eda_histograms_key_features.png"),
        ("Correlação (features clínicas)", FIGURES_DIR / "eda_correlation_key.png"),
        ("Comparação de métricas", FIGURES_DIR / "metrics_comparison.png"),
        ("Curvas ROC", FIGURES_DIR / "roc_curves.png"),
    ]
    for i, (title, path) in enumerate(images):
        with fig_cols[i % 2]:
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)

    st.subheader("Matrizes de confusão")
    cm_cols = st.columns(3)
    for col, name, filename in zip(
        cm_cols,
        ["Regressão logística", "Random Forest", "KNN"],
        [
            "confusion_regressao_logistica.png",
            "confusion_random_forest.png",
            "confusion_knn.png",
        ],
    ):
        path = FIGURES_DIR / filename
        with col:
            if path.exists():
                st.image(str(path), caption=name, use_container_width=True)

    st.subheader("Explicabilidade")
    exp_cols = st.columns(2)
    shap_name = slugify(artifacts["shap_model_name"])
    importance_files = [
        (
            "Importância — regressão logística",
            FIGURES_DIR / "feature_importance_regressao_logistica.png",
        ),
        (
            "Importância — Random Forest",
            FIGURES_DIR / "feature_importance_random_forest.png",
        ),
        (
            f"SHAP global — {artifacts['shap_model_name']}",
            FIGURES_DIR / f"shap_summary_{shap_name}.png",
        ),
        (
            "SHAP local — caso maligno",
            FIGURES_DIR / f"shap_waterfall_{shap_name}_caso_maligno.png",
        ),
    ]
    for i, (title, path) in enumerate(importance_files):
        with exp_cols[i % 2]:
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)

    st.subheader("O modelo pode ser usado na prática?")
    st.markdown(
        """
**Como apoio à triagem, sim; como diagnóstico automático, não.**

- O Wisconsin Diagnostic é pequeno, de uma única fonte, sem validação externa
  multicêntrica.
- As features vêm de núcleos celulares no FNA, não de prontuário eletrônico completo
  (o enunciado também cita violência doméstica — isso exigiria NLP e outro dataset).
- Há colinearidade forte entre raio, perímetro e área; o modelo captura padrões
  estatísticos, não causalidade biológica isolada.
- Qualquer uso real exigiria calibração local, monitoramento de drift, revisão
  regulatória e **supervisão médica obrigatória**.

Na Fase 1 este sistema demonstra o ciclo de ML (EDA → pré-processamento →
modelos → métricas → SHAP) como base de um suporte inteligente à decisão.
"""
    )


def main() -> None:
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    page = st.sidebar.radio(
        "Navegação",
        ["Predição", "Análise do modelo"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Melhor modelo:** {artifacts['best_model_name']}  \n"
        f"**SHAP:** {artifacts['shap_model_name']}  \n"
        f"**Split:** 80/20 estratificado"
    )
    st.sidebar.caption("Tech Challenge IADT — Fase 1")

    if page == "Predição":
        page_prediction(artifacts)
    else:
        page_analysis(artifacts)


main()
