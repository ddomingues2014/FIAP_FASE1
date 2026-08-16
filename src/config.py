"""Caminhos e hiperparâmetros globais do projeto."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Classe positiva = maligno. Recall dessa classe é a métrica principal.
POSITIVE_LABEL = 1
POSITIVE_NAME = "maligno"
NEGATIVE_NAME = "benigno"

ARTIFACTS_PATH = MODELS_DIR / "artifacts.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data"
)
UCI_SOURCE = "sklearn.datasets.load_breast_cancer (UCI Breast Cancer Wisconsin Diagnostic)"

FEATURE_LABELS_PT = {
    "mean_radius": "Raio médio",
    "mean_texture": "Textura média",
    "mean_perimeter": "Perímetro médio",
    "mean_area": "Área média",
    "mean_smoothness": "Suavidade média",
    "mean_compactness": "Compacidade média",
    "mean_concavity": "Concavidade média",
    "mean_concave_points": "Pontos côncavos médios",
    "mean_symmetry": "Simetria média",
    "mean_fractal_dimension": "Dimensão fractal média",
    "radius_error": "Erro do raio",
    "texture_error": "Erro da textura",
    "perimeter_error": "Erro do perímetro",
    "area_error": "Erro da área",
    "smoothness_error": "Erro da suavidade",
    "compactness_error": "Erro da compacidade",
    "concavity_error": "Erro da concavidade",
    "concave_points_error": "Erro dos pontos côncavos",
    "symmetry_error": "Erro da simetria",
    "fractal_dimension_error": "Erro da dimensão fractal",
    "worst_radius": "Pior raio",
    "worst_texture": "Pior textura",
    "worst_perimeter": "Pior perímetro",
    "worst_area": "Pior área",
    "worst_smoothness": "Pior suavidade",
    "worst_compactness": "Pior compacidade",
    "worst_concavity": "Pior concavidade",
    "worst_concave_points": "Piores pontos côncavos",
    "worst_symmetry": "Pior simetria",
    "worst_fractal_dimension": "Pior dimensão fractal",
}

DISCLAIMER = (
    "Esta ferramenta é um **suporte à decisão clínica**, não um diagnóstico. "
    "O modelo foi treinado em um dataset acadêmico pequeno (Wisconsin Diagnostic) "
    "e **não substitui** exame físico, mamografia, biópsia nem a avaliação de um "
    "profissional de saúde. **A palavra final é sempre do médico.**"
)

# Features mais citadas na literatura clínica do FNA para a UI e EDA.
KEY_FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_concavity",
    "mean_concave_points",
    "worst_radius",
    "worst_area",
    "worst_concavity",
    "worst_concave_points",
]


def ensure_directories() -> None:
    """Garante pastas de artefatos usadas no treino e nos relatórios."""
    for path in (RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    """Nome de arquivo ASCII a partir do rótulo do modelo (pt-BR)."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("ç", "c")
        .replace("—", "")
        .replace("(", "")
        .replace(")", "")
        .strip("_")
    )
