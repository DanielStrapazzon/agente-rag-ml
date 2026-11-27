"""
Previsão do modelo Tech Layoffs (etapa 03) — independente do Gemini.

O modelo estima QUANTOS funcionários foram demitidos em um evento de demissão, dado o cenário
informado (país, setor, estágio, porte, capital, ano, mês). NÃO estima a probabilidade de uma
empresa demitir: o dataset só contém eventos que aconteceram.

Este módulo precisa estar importável para carregar modelo/modelo.joblib, porque o
objeto salvo é um EstimadorNaoNegativo (definido aqui, e não em __main__).

Implementado com apoio do Claude Code (Anthropic); revisado e executado pelo grupo.
"""
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from preparar_dados import CATEGORICAS, NUMERICAS

BASE = Path(__file__).resolve().parent
PASTA_MODELO = BASE / "modelo"                # caminho canônico do kit (etapa 04)
ENTRADAS = CATEGORICAS + NUMERICAS          # ordem exata do notebook
UNIDADE_ALVO = "funcionários demitidos no evento"


def limitar_a_zero(previsoes) -> np.ndarray:
    """Pós-processamento do notebook (célula 9): np.maximum(previsão, 0).
    Um evento não pode ter número negativo de demitidos."""
    return np.maximum(np.asarray(previsoes, dtype="float64"), 0)


class EstimadorNaoNegativo(RegressorMixin, BaseEstimator):
    """Envolve o estimador treinado (TransformedTargetRegressor + Pipeline) e aplica
    limitar_a_zero() em predict(). Assim a avaliação de teste, a amostra de conferência e o agente
    usam exatamente a mesma previsão. O pré-processamento continua DENTRO do estimador."""

    def __init__(self, estimador=None):
        self.estimador = estimador

    def fit(self, X, y):
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def _ajustado(self):
        ajustado = getattr(self, "estimador_", None)
        return self.estimador if ajustado is None else ajustado

    def __sklearn_is_fitted__(self):
        return self._ajustado() is not None

    def predict(self, X):
        return limitar_a_zero(self._ajustado().predict(X))


def _vazio(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor)) or \
        (isinstance(valor, str) and valor.strip() == "")


def para_dataframe_modelo(entrada: dict) -> pd.DataFrame:
    """Entrada legível (dict) -> DataFrame de 1 linha no formato do treino:
    colunas na ordem ENTRADAS, categóricas como object (ausente = np.nan), numéricas float64.
    Ausentes (None, "", NaN ou chave faltando) viram np.nan e são imputados pelo Pipeline.
    Não aplica escala nem encoding: isso está dentro do modelo salvo."""
    desconhecidas = sorted(set(entrada) - set(ENTRADAS))
    if desconhecidas:
        raise ValueError(f"atributos não usados pelo modelo: {desconhecidas}; esperados: {ENTRADAS}")
    linha = {}
    for col in CATEGORICAS:
        v = entrada.get(col)
        linha[col] = np.nan if _vazio(v) else str(v)
    for col in NUMERICAS:
        v = entrada.get(col)
        if _vazio(v):
            linha[col] = np.nan
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{col} precisa ser numérico (recebido: {v!r})") from None
        if not math.isfinite(x):
            raise ValueError(f"{col} precisa ser finito (recebido: {v!r})")
        linha[col] = x
    df = pd.DataFrame([linha], columns=ENTRADAS)
    for col in CATEGORICAS:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    for col in NUMERICAS:
        df[col] = df[col].astype("float64")
    return df


def carregar_modelo(caminho: Path | str = PASTA_MODELO / "modelo.joblib"):
    return joblib.load(caminho)


def categorias_conhecidas(modelo) -> dict[str, list[str]]:
    """Categorias vistas no treino, lidas do OneHotEncoder AJUSTADO dentro do modelo."""
    ttr = modelo._ajustado() if isinstance(modelo, EstimadorNaoNegativo) else modelo
    encoder = ttr.regressor_.named_steps["prep"].named_transformers_["cat"].named_steps["encoder"]
    return {col: [str(c) for c in cats] for col, cats in zip(CATEGORICAS, encoder.categories_)}


def valores_imputados(modelo) -> dict:
    """Valor que o Pipeline usa no lugar de um atributo ausente (aprendido só no treino):
    mediana para as numéricas, categoria mais frequente para as categóricas."""
    ttr = modelo._ajustado() if isinstance(modelo, EstimadorNaoNegativo) else modelo
    prep = ttr.regressor_.named_steps["prep"].named_transformers_
    num = prep["num"].named_steps["imputer"].statistics_
    cat = prep["cat"].named_steps["imputer"].statistics_
    return {**{c: float(v) for c, v in zip(NUMERICAS, num)}, **{c: str(v) for c, v in zip(CATEGORICAS, cat)}}


def prever_demissoes(entrada: dict, modelo=None) -> dict:
    """Estimativa do número de demitidos para o cenário informado (formato legível).
    Devolve tipos nativos (JSON). Alertas não alteram a previsão: só a explicam."""
    modelo = modelo if modelo is not None else carregar_modelo()
    X = para_dataframe_modelo(entrada)
    valor = float(modelo.predict(X)[0])
    alertas = []
    for col in ENTRADAS:
        if pd.isna(X.at[0, col]):
            alertas.append(f"{col} não informado: preenchido pelo Pipeline com o valor do treino "
                           f"({'mediana' if col in NUMERICAS else 'categoria mais frequente'})")
    for col, cats in categorias_conhecidas(modelo).items():
        v = X.at[0, col]
        if not pd.isna(v) and v not in cats:
            alertas.append(f"{col} '{v}' não existe no treino: o modelo a ignora (todas as colunas one-hot = 0)")
    porte = X.at[0, "Company_Size_before_Layoffs"]
    if not pd.isna(porte) and valor > porte:
        alertas.append(f"a estimativa ({valor:.0f}) supera o porte anterior informado ({porte:.0f}): "
                       "resultado inconsistente, use com cautela")
    return {"previsao": valor, "unidade": UNIDADE_ALVO,
            "entrada_usada": {c: (None if pd.isna(X.at[0, c]) else
                                  float(X.at[0, c]) if c in NUMERICAS else str(X.at[0, c])) for c in ENTRADAS},
            "alertas": alertas}
