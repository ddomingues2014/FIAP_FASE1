# Checklist de entrega — Fase 1

## Repositório Git

- [x] `git init` (se ainda não existir) e push no GitHub
- [x] Colar a URL do repo no PDF enviado à FIAP e em `reports/relatorio_tecnico.md`
- [ ] README com instruções de execução
- [ ] `Dockerfile` presente
- [ ] Dataset: sklearn + [link Kaggle Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- [ ] Figuras em `reports/figures/`
- [ ] Relatório técnico (`reports/relatorio_tecnico.md` e `.pdf`)

## Como rodar (para gravar o vídeo)

```bash
python3 -m virtualenv .venv   # ou python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/explore.py
python -m src.train
streamlit run app/streamlit_app.py
```

## Roteiro do vídeo (≤ 15 min)

1. (1–2 min) Contexto + recorte Wisconsin + disclaimer
2. (3 min) `python scripts/explore.py` — EDA, balanceamento, correlação
3. (4 min) `python -m src.train` — três modelos, recall, SHAP
4. (4 min) Streamlit — caso maligno, caso benigno, waterfall
5. (1 min) Limitações e papel do médico + link do GitHub

## PDF da FIAP

O arquivo enviado à plataforma deve conter: link do Git, prints, relatório técnico e o link do vídeo (YouTube/Vimeo, público ou não listado).
