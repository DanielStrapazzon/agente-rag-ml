"""
EXEMPLO DO "CASO 2": transformações feitas FORA do Pipeline (como em muitos notebooks).
Rodem a partir da pasta do kit:   python exemplo_transformacao/treinar_fora_do_pipeline.py

Aqui o dataset legível (dados/dataset_legivel.csv) passa por:
  1. bairro (texto) -> colunas 0/1 com pd.get_dummies
  2. area_m2, quartos, vagas -> StandardScaler (ajustado SÓ no treino)
e o modelo é treinado nesses dados transformados.

O que NÃO pode ser esquecido: salvar os transformadores AJUSTADOS (scaler, lista de colunas,
mapas de códigos) para aplicar a MESMA transformação na entrada do usuário (para_o_modelo).
"""

# --- Etapa 05 (grupo): proteção dos artefatos Tech Layoffs. Este EXEMPLO grava em dados/ e modelo/;
# ele só roda se esses arquivos não forem do Tech Layoffs ou com PERMITIR_SOBRESCREVER_ARTEFATOS=1.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from protecao_artefatos import exigir_permissao_para_sobrescrever  # noqa: E402
exigir_permissao_para_sobrescrever(_Path(__file__).name)
# --- fim da proteção

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

os.chdir(Path(__file__).resolve().parent.parent)
NUMERICAS, CATEGORICAS, ALVO = ["area_m2", "quartos", "vagas"], ["bairro"], "aluguel"

df = pd.read_csv("dados/dataset_legivel.csv")                  # dados LEGÍVEIS (iguais ao do RAG)
X, y = df[NUMERICAS + CATEGORICAS], df[ALVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


def transformar(X_legivel: pd.DataFrame, scaler: StandardScaler, colunas: list[str]) -> pd.DataFrame:
    """A MESMA transformação usada no treino, no teste e (depois) no agente."""
    Xt = pd.get_dummies(X_legivel, columns=CATEGORICAS, dtype=int)  # texto -> 0/1
    Xt = Xt.reindex(columns=colunas, fill_value=0)                  # mesmas colunas, mesma ordem
    Xt[NUMERICAS] = scaler.transform(Xt[NUMERICAS])                 # mesma escala do treino
    return Xt


scaler = StandardScaler().fit(X_tr[NUMERICAS])                     # ajusta SÓ no treino
colunas = list(pd.get_dummies(X_tr, columns=CATEGORICAS, dtype=int).columns)
modelo = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(transformar(X_tr, scaler, colunas), y_tr)
prev = modelo.predict(transformar(X_te, scaler, colunas))

joblib.dump(modelo, "modelo/modelo.joblib")                         # só o estimador
joblib.dump({"scaler": scaler, "colunas_modelo": colunas}, "modelo/transformadores.joblib")   # <- NÃO ESQUECER
json.dump({"modelo": "KNN (k=7, weights=distance), transformações fora do Pipeline",
           "r2_teste": round(float(r2_score(y_te, prev)), 3), "mae_teste": round(float(mean_absolute_error(y_te, prev)), 2),
           "sklearn_version": sklearn.__version__},
          open("modelo/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
amostra = [{"entrada": {k: (v.item() if hasattr(v, "item") else v) for k, v in linha.items()}, "previsao_no_treino": float(p)}
           for (_, linha), p in zip(X_te.head(5).iterrows(), prev[:5])]
json.dump(amostra, open("modelo/amostra_conferencia.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok: modelo.joblib, transformadores.joblib, metricas.json e amostra_conferencia.json gerados")
