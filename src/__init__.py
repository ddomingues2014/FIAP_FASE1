"""Pipeline de classificação de câncer de mama (Wisconsin) — Tech Challenge IADT Fase 1."""

from .config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from .data import load_wisconsin, stratified_split

__all__ = [
    "FIGURES_DIR",
    "MODELS_DIR",
    "RANDOM_STATE",
    "TEST_SIZE",
    "load_wisconsin",
    "stratified_split",
]
