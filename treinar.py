"""
Treino do modelo Tech Layoffs (etapa 03) — reproduz as células 7 e 9 de notebooks/analise_modelos.ipynb.

    .venv/bin/python treinar.py

Gera (caminhos canônicos do kit desde a etapa 04):
    modelo/modelo.joblib            melhor estimador (pré-processamento + log1p + corte em 0)
    modelo/metricas.json            métricas reais, versões, hashes
    modelo/amostra_conferencia.json entradas legíveis do teste + previsão do treino
    reports/novo_treinamento/metricas_teste.csv, melhores_parametros.csv

Seleção do modelo: menor MAE médio da validação cruzada (o teste não participa da escolha).
Implementado com apoio do Claude Code (Anthropic); revisado e executado pelo grupo.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preparar_dados import (ALVO, CATEGORICAS, DESTINO as CSV_LEGIVEL, EXCLUIDAS_VAZAMENTO, NUMERICAS,
                            ORIGEM, preparar, separar_atributos_modelo)
from previsao_layoffs import ENTRADAS, PASTA_MODELO, UNIDADE_ALVO, EstimadorNaoNegativo, limitar_a_zero

BASE = Path(__file__).resolve().parent
PASTA_RELATORIO = BASE / "reports" / "novo_treinamento"
SEED = 42
TAMANHO_TESTE = 0.2
N_AMOSTRA = 5


def sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def carregar_xy() -> tuple[pd.DataFrame, pd.Series, dict]:
    """X/y a partir da preparação da etapa 02 (mesmos registros do notebook)."""
    df, resumo = preparar(ORIGEM)
    X, y = separar_atributos_modelo(df)
    return X, y, resumo


def conferir_csv_legivel(X: pd.DataFrame, y: pd.Series) -> bool | None:
    """True se o CSV legível gera exatamente os mesmos X/y; None se o CSV ainda não existe."""
    if not CSV_LEGIVEL.exists():
        return None
    Xl, yl = separar_atributos_modelo(pd.read_csv(CSV_LEGIVEL, low_memory=False))
    try:
        pd.testing.assert_frame_equal(X.reset_index(drop=True), Xl)
        pd.testing.assert_series_equal(y.reset_index(drop=True), yl)
    except AssertionError:
        return False
    return True


def criar_preprocessamento() -> ColumnTransformer:
    """Igual à célula 7 do notebook."""
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERICAS),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAS),
    ])


def modelos_e_grades() -> dict:
    """Igual à célula 9 do notebook."""
    return {
        "Regressão Linear Múltipla": (LinearRegression(), {}),
        "KNN": (KNeighborsRegressor(), {
            "model__n_neighbors": [3, 7, 15],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]}),
        "MLP": (MLPRegressor(random_state=SEED, max_iter=400, early_stopping=True, n_iter_no_change=15), {
            "model__hidden_layer_sizes": [(32,), (64, 32)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.001, 0.1]}),
    }


def criar_busca(modelo, params: dict) -> GridSearchCV:
    estimador = TransformedTargetRegressor(regressor=Pipeline([("prep", criar_preprocessamento()), ("model", modelo)]),
                                           func=np.log1p, inverse_func=np.expm1)
    grade = {f"regressor__{k}": v for k, v in params.items()}
    return GridSearchCV(estimador, grade or [{}], cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                        scoring="neg_mean_absolute_error", n_jobs=1, refit=True, error_score="raise")


def metricas(y_real, previsto) -> dict:
    return {"R2": float(r2_score(y_real, previsto)),
            "RMSE": float(np.sqrt(mean_squared_error(y_real, previsto))),
            "MAE": float(mean_absolute_error(y_real, previsto))}


def treinar_todos(X_tr, y_tr, X_te, y_te) -> list[dict]:
    resultados = []
    for nome, (modelo, params) in modelos_e_grades().items():
        print(f"Treinando: {nome}", flush=True)
        inicio = time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            gs = criar_busca(modelo, params).fit(X_tr, y_tr)
        final = EstimadorNaoNegativo(gs.best_estimator_)
        prev = final.predict(X_te)                     # = np.maximum(gs.predict(X_te), 0), como no notebook
        contagem = {}
        for a in avisos:
            contagem[a.category.__name__] = contagem.get(a.category.__name__, 0) + 1
        r = {"Modelo": nome, **metricas(y_te, prev), "MAE_CV": float(-gs.best_score_),
             "Valores testados": str(params) if params else "sem grade (baseline)",
             "Melhores parâmetros": str(gs.best_params_) if params else "padrão",
             "melhores_parametros": {k.replace("regressor__model__", ""): v for k, v in gs.best_params_.items()},
             "segundos": round(time.perf_counter() - inicio, 1), "avisos": contagem,
             "acima_do_porte": int((prev > X_te["Company_Size_before_Layoffs"].to_numpy()).sum()),
             "estimador": final, "previsoes_teste": prev}
        print(f"  MAE CV {r['MAE_CV']:.2f} | teste: MAE {r['MAE']:.2f} RMSE {r['RMSE']:.2f} R² {r['R2']:.4f} "
              f"| {r['segundos']} s | avisos {contagem or 'nenhum'}", flush=True)
        resultados.append(r)
    return resultados


def selecionar(resultados: list[dict]) -> dict:
    """Menor MAE médio de validação cruzada."""
    return min(resultados, key=lambda r: r["MAE_CV"])


def descrever(r: dict) -> str:
    if not r["melhores_parametros"]:
        return r["Modelo"]
    return f"{r['Modelo']} (" + ", ".join(f"{k}={v}" for k, v in r["melhores_parametros"].items()) + ")"


def _legivel(valor, col: str):
    if pd.isna(valor):
        return None
    if col in CATEGORICAS:
        return str(valor)
    x = float(valor)
    return int(x) if x.is_integer() else x


def amostra_conferencia(X_te: pd.DataFrame, previsoes: np.ndarray, n: int = N_AMOSTRA) -> list[dict]:
    """As n primeiras linhas do teste + a 1ª linha do teste com ausente (se ainda não incluída),
    em formato legível, com a previsão obtida no treino."""
    indices = list(range(min(n, len(X_te))))
    com_ausente = np.flatnonzero(X_te.isna().any(axis=1).to_numpy())
    if len(com_ausente) and not any(X_te.iloc[i].isna().any() for i in indices):
        indices.append(int(com_ausente[0]))
    return [{"entrada": {c: _legivel(X_te.iloc[i][c], c) for c in ENTRADAS},
             "previsao_no_treino": float(previsoes[i])} for i in indices]


def montar_metricas(sel: dict, resultados: list[dict], n_tr: int, n_te: int, resumo: dict, csv_ok) -> dict:
    return {
        "modelo": descrever(sel),
        "tarefa": "regressao",
        "alvo": ALVO,
        "unidade_alvo": UNIDADE_ALVO,
        "r2_teste": round(sel["R2"], 4),
        "mae_teste": round(sel["MAE"], 2),
        "rmse_teste": round(sel["RMSE"], 2),
        "mae_cv": round(sel["MAE_CV"], 2),
        "melhores_parametros": sel["melhores_parametros"],
        "criterio_selecao": "menor MAE médio na validação cruzada (KFold 5, só treino)",
        "entradas_categoricas": CATEGORICAS,
        "entradas_numericas": NUMERICAS,
        "excluidas_por_vazamento": EXCLUIDAS_VAZAMENTO,
        "pos_processamento": "previsão = max(expm1(modelo(log1p)), 0) — EstimadorNaoNegativo em previsao_layoffs.py",
        "random_state": SEED,
        "tamanho_teste": TAMANHO_TESTE,
        "quantidade_treino": n_tr,
        "quantidade_teste": n_te,
        "previsoes_teste_acima_do_porte_anterior": sel["acima_do_porte"],
        "comparacao_modelos": [{"modelo": r["Modelo"], "mae_cv": round(r["MAE_CV"], 2),
                                "mae_teste": round(r["MAE"], 2), "rmse_teste": round(r["RMSE"], 2),
                                "r2_teste": round(r["R2"], 4), "melhores_parametros": r["melhores_parametros"]}
                               for r in resultados],
        "dataset": {"arquivo": "dados/tech_layoffs.csv", "sha256": sha256(ORIGEM),
                    "linhas_preparadas": resumo["linhas_finais"],
                    "csv_legivel_sha256": sha256(CSV_LEGIVEL) if CSV_LEGIVEL.exists() else None,
                    "csv_legivel_equivalente": csv_ok},
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o))


def exportar(sel: dict, resultados: list[dict], amostra: list[dict], met: dict,
             pasta_modelo: Path = PASTA_MODELO, pasta_relatorio: Path = PASTA_RELATORIO) -> None:
    pasta_modelo.mkdir(parents=True, exist_ok=True)
    pasta_relatorio.mkdir(parents=True, exist_ok=True)
    joblib.dump(sel["estimador"], pasta_modelo / "modelo.joblib")
    (pasta_modelo / "metricas.json").write_text(_json(met) + "\n", encoding="utf-8")
    (pasta_modelo / "amostra_conferencia.json").write_text(_json(amostra) + "\n", encoding="utf-8")
    tabela = pd.DataFrame(resultados)
    tabela[["Modelo", "R2", "RMSE", "MAE"]].sort_values("MAE").to_csv(pasta_relatorio / "metricas_teste.csv", index=False)
    (tabela[["Modelo", "Valores testados", "Melhores parâmetros", "MAE_CV"]]
     .rename(columns={"MAE_CV": "MAE médio CV"}).to_csv(pasta_relatorio / "melhores_parametros.csv", index=False))


def validar_em_novo_processo(pasta_modelo: Path = PASTA_MODELO) -> str:
    """Carrega o .joblib num processo Python novo (sem nada deste script em memória) e refaz a amostra."""
    codigo = (
        "import json,sys; sys.path.insert(0, sys.argv[1])\n"
        "from pathlib import Path\n"
        "from previsao_layoffs import carregar_modelo, para_dataframe_modelo\n"
        "p = Path(sys.argv[2]); m = carregar_modelo(p / 'modelo.joblib')\n"
        "erros = 0\n"
        "for item in json.loads((p / 'amostra_conferencia.json').read_text(encoding='utf-8')):\n"
        "    obtido = float(m.predict(para_dataframe_modelo(item['entrada']))[0])\n"
        "    esperado = item['previsao_no_treino']\n"
        "    erros += abs(obtido - esperado) > 1e-9 * max(1.0, abs(esperado))\n"
        "print(f'{type(m).__name__}: amostra reproduzida, divergencias={erros}')\n"
        "sys.exit(1 if erros else 0)\n")
    r = subprocess.run([sys.executable, "-c", codigo, str(BASE), str(pasta_modelo)],
                       capture_output=True, text=True, cwd="/")
    if r.returncode != 0:
        raise RuntimeError(f"validação em novo processo falhou:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def main() -> dict:
    print(f"Python {platform.python_version()} | scikit-learn {sklearn.__version__} | numpy {np.__version__} "
          f"| pandas {pd.__version__} | joblib {joblib.__version__}")
    X, y, resumo = carregar_xy()
    assert list(X.columns) == ENTRADAS and not set(EXCLUIDAS_VAZAMENTO) & set(X.columns)
    csv_ok = conferir_csv_legivel(X, y)
    print(f"Dados: {len(X)} linhas, {X.shape[1]} entradas | CSV legível equivalente: {csv_ok}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TAMANHO_TESTE, random_state=SEED)
    print(f"Treino: {len(X_tr)}, teste: {len(X_te)}")
    resultados = treinar_todos(X_tr, y_tr, X_te, y_te)
    sel = selecionar(resultados)
    menor_teste = min(resultados, key=lambda r: r["MAE"])["Modelo"]
    print(f"Selecionado (menor MAE CV): {descrever(sel)} | menor MAE de teste: {menor_teste}")
    amostra = amostra_conferencia(X_te, sel["previsoes_teste"])
    met = montar_metricas(sel, resultados, len(X_tr), len(X_te), resumo, csv_ok)
    exportar(sel, resultados, amostra, met)
    print("Novo processo:", validar_em_novo_processo())
    print(f"Artefatos em {PASTA_MODELO.relative_to(BASE)}/ e {PASTA_RELATORIO.relative_to(BASE)}/")
    return met


if __name__ == "__main__":
    main()
