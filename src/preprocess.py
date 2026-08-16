"""Pipeline de pré-processamento para features numéricas do FNA."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocess_pipeline() -> Pipeline:
    """Imputação + padronização.

    O Wisconsin via sklearn praticamente não tem missing, mas o imputador
    documenta o tratamento exigido pelo enunciado e deixa o pipeline robusto
    caso o CSV do Kaggle (com coluna `Unnamed: 32`/`id`) seja usado no futuro.

    StandardScaler é obrigatório para KNN e ajuda a regressão logística.
    Árvores não precisam de escala, mas manter o mesmo pré-processamento
    simplifica a comparação justa entre modelos.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_model_pipeline(estimator) -> Pipeline:
    """Encadeia pré-processamento e classificador sem vazar estatísticas do teste."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocess_pipeline()),
            ("model", estimator),
        ]
    )
