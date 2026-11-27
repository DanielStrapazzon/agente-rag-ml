"""
Gera o "gabarito" do agente direto do CSV e do modelo do repositório (Tech Layoffs, regressão).
O professor roda este script no commit entregue para conferir os números da bateria.
    python conferir.py                  -> estatísticas + previsões das 3 primeiras linhas do dataset
    python conferir.py entradas.csv     -> previsões para as linhas de um CSV com as 8 colunas de entrada
                                           (recomendado: use casos NOVOS, que não estão no dataset)
Funciona sem chave do Gemini. Adaptado ao Tech Layoffs na etapa 04 (com apoio do Claude Code).
"""
import sys

import pandas as pd

from ferramentas import (ALVO, CATEGORICAS, DF, METRICAS, MODELO, NUMERICAS, conferir_transformacao,
                         consultar_dados, para_o_modelo)

pd.set_option("display.width", 200, "display.max_columns", None)
print("== Métricas salvas:", {k: METRICAS[k] for k in ("modelo", "r2_teste", "mae_teste", "rmse_teste", "mae_cv",
                                                       "sklearn_version")})
print("== Conferência da ponte legível -> modelo:", conferir_transformacao())
print(f"== Dataset: {DF.shape[0]} linhas x {DF.shape[1]} colunas | alvo: {ALVO} | tarefa: regressão")
print(f"== Eventos: {len(DF)} | empresas distintas: {DF['Company'].nunique()} | "
      f"funcionários demitidos (soma de {ALVO}): {int(DF[ALVO].sum())}")
print(f"== Período: {DF['Date_layoffs'].min()} a {DF['Date_layoffs'].max()}")
print("\n== Estatísticas numéricas\n", DF[NUMERICAS + [ALVO]].describe().round(2))
print("\n== Por ano (year)\n", DF.groupby("year")[ALVO].agg(eventos="count", demitidos="sum", media="mean",
                                                            mediana="median").round(2))
for col in CATEGORICAS:
    tab = DF.groupby(col, dropna=False)[ALVO].agg(eventos="count", demitidos="sum", media="mean", mediana="median",
                                                  minimo="min", maximo="max").round(2)
    print(f"\n== {ALVO} por {col} (15 maiores em demitidos; grafias originais)\n",
          tab.sort_values("demitidos", ascending=False).head(15))
print("\n== Maiores eventos\n", DF.nlargest(5, ALVO)[["Company", "Date_layoffs", "Country", "Industry", ALVO]])
print("\n== Correlação com o alvo (Spearman)\n",
      DF[NUMERICAS + [ALVO]].corr(method="spearman")[ALVO].drop(ALVO).round(3))
print("\n== Exemplo de consulta estruturada (Brasil, 2023):",
      consultar_dados({"operacao": "soma", "coluna": ALVO, "filtros": [
          {"coluna": "Country", "operador": "eq", "valor": "Brazil"}, {"coluna": "year", "operador": "eq", "valor": 2023}]}))

entradas = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else DF.head(3)
entradas = entradas[CATEGORICAS + NUMERICAS]
print("\n== Previsões do modelo (estimativa de demitidos no evento; não é probabilidade de demissão)")
saida = entradas.copy()
X_modelo = pd.concat([para_o_modelo(e) for e in entradas.to_dict("records")], ignore_index=True)   # PONTE
saida["previsao"] = MODELO.predict(X_modelo).round(2)
saida["acima_do_porte"] = saida["previsao"] > saida["Company_Size_before_Layoffs"]
if len(sys.argv) == 1:
    saida[f"{ALVO}_real"] = DF.head(3)[ALVO].to_numpy()
print(saida.to_string())
