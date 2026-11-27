"""
EXEMPLO DE CLASSIFICAÇÃO (dados sintéticos de inadimplência).
Rodem a partir da pasta do kit:   python exemplo_classificacao/treinar_classificacao.py
Ele SOBRESCREVE dados/dataset_legivel.csv e modelo/ com a versão de classificação.

O que importa copiar para o trabalho de vocês: o Pipeline com ColumnTransformer
(o .joblib aceita as colunas ORIGINAIS, com texto) e o joblib.dump(grid.best_estimator_).
"""

# --- Etapa 05 (grupo): proteção dos artefatos Tech Layoffs. Este EXEMPLO grava em dados/ e modelo/;
# ele só roda se esses arquivos não forem do Tech Layoffs ou com PERMITIR_SOBRESCREVER_ARTEFATOS=1.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from protecao_artefatos import exigir_permissao_para_sobrescrever  # noqa: E402
exigir_permissao_para_sobrescrever(_Path(__file__).name)
# --- fim da proteção

from pathlib import Path
import json, os, sys
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # só para o exemplo ficar limpo

os.chdir(Path(__file__).resolve().parent.parent)          # trabalha na pasta do kit
ALVO_NUM = len(sys.argv) > 1 and sys.argv[1] == "num"     # variante: alvo 0/1 em vez de "sim"/"nao"
rng = np.random.default_rng(42); n = 1200
esc = ["Fundamental", "Médio", "Superior", "Pós-graduação"]
df = pd.DataFrame({
    "limite_credito": rng.uniform(500, 50000, n).round(0),
    "idade": rng.integers(18, 76, n),
    "atrasos_6m": rng.integers(0, 7, n),
    "renda_mensal": rng.uniform(1200, 25000, n).round(0),
    "escolaridade": rng.choice(esc, n),
})
z = -2.2 + 0.75*df["atrasos_6m"] - 0.00004*df["limite_credito"] - 0.00006*df["renda_mensal"] + 0.4*(df["escolaridade"]=="Fundamental")
p = 1/(1+np.exp(-z))
y01 = (rng.uniform(0, 1, n) < p).astype(int)
df["inadimplente"] = y01 if ALVO_NUM else np.where(y01 == 1, "sim", "nao")
df.to_csv("dados/dataset_legivel.csv", index=False)

X, y = df.drop(columns="inadimplente"), df["inadimplente"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
prep = ColumnTransformer([("num", StandardScaler(), ["limite_credito", "idade", "atrasos_6m", "renda_mensal"]),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), ["escolaridade"])])
pipe = Pipeline([("prep", prep), ("clf", MLPClassifier(max_iter=600, random_state=42))])
grid = GridSearchCV(pipe, {"clf__hidden_layer_sizes": [(16,), (32, 16)], "clf__alpha": [1e-4, 1e-2]}, cv=5, scoring="f1_macro")
grid.fit(X_tr, y_tr)
melhor = grid.best_estimator_
prev = melhor.predict(X_te)
pos = 1 if ALVO_NUM else "sim"
metricas = {"modelo": f"MLPClassifier {grid.best_params_}",
            "acuracia_teste": round(float(accuracy_score(y_te, prev)), 3),
            "f1_teste": round(float(f1_score(y_te, prev, pos_label=pos)), 3),
            "sklearn_version": sklearn.__version__}
joblib.dump(melhor, "modelo/modelo.joblib")
MODELO_FINAL = melhor
# Amostra para conferir a PONTE legível -> modelo (ferramentas.conferir_transformacao):
# 5 linhas de teste em formato LEGÍVEL + a previsão que o modelo deu aqui, no treino.
amostra = [{"entrada": {k: (v.item() if hasattr(v, "item") else v) for k, v in linha.items()},
            "previsao_no_treino": (p.item() if hasattr(p, "item") else p)}
           for (_, linha), p in zip(X_te.head(5).iterrows(), MODELO_FINAL.predict(X_te.head(5)))]
json.dump(amostra, open("modelo/amostra_conferencia.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

json.dump(metricas, open("modelo/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(metricas, df["inadimplente"].value_counts().to_dict())
