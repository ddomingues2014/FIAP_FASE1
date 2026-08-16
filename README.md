# Tech Challenge IADT Fase 1 — suporte ao diagnóstico de câncer de mama

Repositório: https://github.com/ddomingues2014/FIAP_FASE1


Sistema inicial de IA (Machine Learning tabular) para **classificar tumores de mama em maligno ou benigno** a partir das medidas do exame FNA do dataset **Breast Cancer Wisconsin Diagnostic** (UCI / Kaggle).

Projeto em **scripts Python** (sem notebook). O enunciado aceita “Notebook Jupyter ou scripts Python”.

> **Disclaimer clínico:** ferramenta de **suporte à decisão**. Não substitui mamografia, biópsia nem a avaliação médica. **A palavra final é sempre do médico.**

Projeto em **scripts Python** (sem notebook). O enunciado aceita “Notebook Jupyter ou scripts Python”.

> **Disclaimer clínico:** ferramenta de **suporte à decisão**. Não substitui mamografia, biópsia nem a avaliação médica. **A palavra final é sempre do médico.**

## Problema

Uma rede de hospitais especializados na saúde da mulher precisa acelerar a triagem de risco. Nesta Fase 1 o recorte é diagnóstico de câncer de mama com dados estruturados:

- **Classe positiva (1):** maligno
- **Classe negativa (0):** benigno
- **Métrica principal:** recall da classe maligna (falso negativo é o erro mais grave), depois F1 e matriz de confusão

Dataset equivalente ao sugerido no enunciado:
[Breast Cancer Wisconsin (Diagnostic) no Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

O projeto carrega a mesma base via `sklearn.datasets.load_breast_cancer` (reproduzível, sem download manual).

## Estrutura

```text
tech_fiap/
├── README.md
├── requirements.txt
├── Dockerfile
├── app/streamlit_app.py          # demo do vídeo
├── src/                          # pipeline reprodutível
│   ├── data.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── scripts/
│   ├── explore.py                # EDA + discussão no terminal
│   ├── export_figures.py         # EDA + treino + todas as figuras
│   └── build_report.py
├── models/                       # artefatos .joblib + métricas
├── reports/
│   ├── figures/                  # prints para o PDF da FIAP
│   ├── relatorio_tecnico.md
│   └── relatorio_tecnico.pdf
└── data/raw/                     # placeholder (dados via sklearn)
```

## Como executar

Python 3.10+ recomendado (3.11 no Docker).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se `python3 -m venv` falhar no Ubuntu (`ensurepip` ausente), use:

```bash
pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Exploração de dados

```bash
python scripts/explore.py
# equivalente:
python -m src.eda
```

### 2. Treinar modelos e gerar figuras

```bash
python -m src.train
# ou EDA + treino + todas as figuras:
python scripts/export_figures.py
```

Isso grava `models/artifacts.joblib`, métricas e PNGs em `reports/figures/`.

### 3. App Streamlit (demonstração)

```bash
streamlit run app/streamlit_app.py
```

Telas:

- **Predição:** caso de teste ou valores manuais + probabilidade + SHAP local
- **Análise do modelo:** EDA, comparação de métricas, matrizes, SHAP global

### 4. Relatório PDF

```bash
python scripts/build_report.py
```

Gera `reports/relatorio_tecnico.pdf` a partir de `reports/relatorio_tecnico.md`.

## Docker

```bash
docker build -t iadt-fase1 .
docker run --rm -p 8501:8501 iadt-fase1
```

Abra `http://localhost:8501`. A imagem treina na build se `models/artifacts.joblib` ainda não existir.

## Modelos

| Modelo | Por quê |
|---|---|
| Regressão logística | Baseline clínico interpretável (coeficientes) |
| Random Forest | Não-linearidade + feature importance nativa |
| KNN | Terceiro classificador simples; depende fortemente da escala |

Todos passam por `Pipeline` (`SimpleImputer` + `StandardScaler` + classificador), com **split 80/20 estratificado** e `GridSearchCV` leve otimizando **recall**.

## Explicabilidade

- Importância de features (Gini no RF e \|coef\| na logística)
- SHAP global (beeswarm) e local (waterfall) em casos maligno e benigno

## Entregáveis da Fase 1

- Código-fonte neste repositório
- Este README
- Dockerfile
- Dataset via sklearn + [link Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- Figuras em `reports/figures/`
- Relatório técnico em `reports/relatorio_tecnico.md` / `.pdf`
- Vídeo de até 15 min (YouTube/Vimeo) — roteiro no relatório
- Checklist de entrega: [`reports/checklist_entrega.md`](reports/checklist_entrega.md)

## Limitações

Dataset único e pequeno (~569 amostras), sem validação externa, com colinearidade entre raio/perímetro/área. Não cobre NLP de prontuário (ex.: violência doméstica citada no enunciado). Uso prático só como **apoio**, com o médico no loop.
