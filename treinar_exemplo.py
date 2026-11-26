"""
EXEMPLO DIDÁTICO (opcional: modelo/ e dados/ já vêm prontos no kit) — gera um dataset sintético de aluguel de imóveis e treina um
Pipeline (pré-processamento + KNN) só para o kit funcionar "de fábrica".

No trabalho de vocês, SUBSTITUAM este script pelo código do Trabalho Prático
anterior: o mesmo dataset limpo, o mesmo Pipeline e o melhor modelo encontrado
no GridSearchCV, salvo com joblib.dump(...) em modelo/modelo.joblib.
"""
import json
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

SEED = 42
rng = np.random.default_rng(SEED)
n = 1500
bairros = ["Centro", "Efapi", "Passo dos Fortes", "Presidente Médici", "Jardim Itália"]
fator = {"Centro": 1.35, "Efapi": 0.85, "Passo dos Fortes": 1.0, "Presidente Médici": 0.9, "Jardim Itália": 1.2}

df = pd.DataFrame({
    "area_m2": rng.uniform(30, 220, n).round(0),
    "quartos": rng.integers(1, 5, n),
    "vagas": rng.integers(0, 3, n),
    "bairro": rng.choice(bairros, n),
})
df["aluguel"] = (
    (18 * df["area_m2"] + 250 * df["quartos"] + 180 * df["vagas"]) * df["bairro"].map(fator)
    + rng.normal(0, 250, n)
).round(0)

df.to_csv("dados/dataset_legivel.csv", index=False)

X, y = df.drop(columns="aluguel"), df["aluguel"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

prep = ColumnTransformer([
    ("num", StandardScaler(), ["area_m2", "quartos", "vagas"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["bairro"]),
])
modelo = Pipeline([("prep", prep), ("reg", KNeighborsRegressor(n_neighbors=7, weights="distance"))])
modelo.fit(X_tr, y_tr)

prev = modelo.predict(X_te)
metricas = {
    "modelo": "KNN (k=7, weights=distance)",
    "r2_teste": round(float(r2_score(y_te, prev)), 3),
    "mae_teste": round(float(mean_absolute_error(y_te, prev)), 2),
    "sklearn_version": sklearn.__version__,
}
joblib.dump(modelo, "modelo/modelo.joblib")
MODELO_FINAL = modelo
# Amostra para conferir a PONTE legível -> modelo (ferramentas.conferir_transformacao):
# 5 linhas de teste em formato LEGÍVEL + a previsão que o modelo deu aqui, no treino.
amostra = [{"entrada": {k: (v.item() if hasattr(v, "item") else v) for k, v in linha.items()},
            "previsao_no_treino": (p.item() if hasattr(p, "item") else p)}
           for (_, linha), p in zip(X_te.head(5).iterrows(), MODELO_FINAL.predict(X_te.head(5)))]
json.dump(amostra, open("modelo/amostra_conferencia.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

json.dump(metricas, open("modelo/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(metricas)
