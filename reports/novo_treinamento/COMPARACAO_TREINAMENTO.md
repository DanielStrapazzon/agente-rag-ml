# Comparação: treino original (Colab) × novo treino (`treinar.py`)

> Números obtidos executando `.venv/bin/python treinar.py` em 2026-11-26 (duas execuções com
> resultados idênticos). Os resultados originais estão preservados em `reports/resultados/`.
> Script e documento produzidos com apoio do Claude Code (Anthropic); execução e revisão pelo grupo.

## 1. Metodologia reproduzida

Idêntica às células 7 e 9 de `notebooks/analise_modelos.ipynb`:

- **Dados:** `dados/tech_layoffs.csv` preparado por `preparar_dados.py` (mesma limpeza do notebook):
  1.745 eventos, nenhum removido. As entradas obtidas da preparação e do CSV legível
  `dados/dataset_legivel_tech_layoffs.csv` são idênticas (conferido em `treinar.py`).
- **Entradas (8):** `Country`, `Continent`, `Industry`, `Stage` (categóricas);
  `Company_Size_before_Layoffs`, `Money_Raised_clean`, `year`, `month` (numéricas).
  **Alvo:** `Laid_Off`. **Excluídas por vazamento:** `Percentage`, `Company_Size_after_layoffs`.
- **Divisão:** 80/20, `random_state=42` → 1.396 treino / 349 teste.
- **Pré-processamento (dentro do Pipeline, ajustado só no treino de cada fold):** numéricas =
  mediana + `StandardScaler`; categóricas = mais frequente + `OneHotEncoder(handle_unknown="ignore")`.
- **Alvo:** `TransformedTargetRegressor(func=log1p, inverse_func=expm1)`.
- **Busca:** `GridSearchCV`, `KFold(5, shuffle=True, random_state=42)`, `neg_mean_absolute_error`, `n_jobs=1`.
- **Grades:** Regressão Linear sem grade; KNN `n_neighbors` [3, 7, 15] × `weights` [uniform, distance]
  × `metric` [euclidean, manhattan]; MLP (`max_iter=400`, `early_stopping=True`,
  `n_iter_no_change=15`, `random_state=42`) `hidden_layer_sizes` [(32,), (64, 32)] × `activation`
  [relu, tanh] × `alpha` [0.001, 0.1].
- **Pós-processamento:** previsões de teste limitadas a ≥ 0, como no notebook. O MAE de CV usa a
  previsão sem corte, também como no notebook.

## 2. Versões

| | Treino original | Novo treino |
|---|---|---|
| Ambiente | Google Colab | `.venv` local (Fedora) |
| Python | não registrado | 3.14.10 |
| scikit-learn | não registrado | 1.9.1 |
| numpy | não registrado | 2.5.3 |
| pandas | não registrado | 3.0.5 |
| joblib | não registrado | 1.6.0 |

## 3. Resultados

| Modelo | MAE CV | MAE teste | RMSE teste | R² teste | Melhores parâmetros |
|---|---:|---:|---:|---:|---|
| **MLP** | **201,19** | **236,03** | **851,53** | **0,5537** | activation=tanh, alpha=0.1, hidden_layer_sizes=(64, 32) |
| KNN | 210,19 | 258,29 | 884,34 | 0,5186 | metric=euclidean, n_neighbors=7, weights=distance |
| Regressão Linear Múltipla | 101.025,85 | 884,24 | 8.331,71 | −41,7277 | padrão |

Tempo: Regressão Linear 1,9 s; KNN 1,9 s; MLP 17,0 s. **Nenhum aviso** (inclusive
`ConvergenceWarning`) foi emitido em nenhum algoritmo.

## 4. Diferenças em relação ao treino original

| Modelo | Δ R² | Δ RMSE | Δ MAE | Δ MAE CV | Parâmetros |
|---|---:|---:|---:|---:|---|
| MLP | −1,1 × 10⁻¹⁶ | +1,1 × 10⁻¹³ | −8,5 × 10⁻¹⁴ | 0 | iguais |
| KNN | 0 | 0 | 0 | 0 | iguais |
| Regressão Linear | 0 | 0 | 0 | 0 | iguais |

O novo treino **reproduziu os resultados originais**. As únicas diferenças estão na ordem de 10⁻¹³,
erro de arredondamento de ponto flutuante. As grades testadas e os melhores parâmetros são os mesmos.

## 5. Modelo selecionado: MLP

- **Critério:** menor MAE médio na validação cruzada, calculado só com o treino: MLP 201,19 contra
  KNN 210,19 e Regressão Linear 101.025,85. O teste não foi usado para escolher.
- O notebook escolhia pelo menor MAE de **teste**. Aqui os dois critérios apontam o mesmo modelo (MLP),
  então a escolha não muda.
- **Ressalva:** a vantagem do MLP sobre o KNN é pequena (9,0 no MAE de CV; 22,3 no MAE de teste;
  0,035 no R²). Com uma única divisão treino/teste, não dá para afirmar que o MLP é melhor de forma
  robusta. O KNN segue como alternativa razoável.
- **Regressão Linear:** o R² negativo e o MAE de CV enorme vêm da combinação de modelo linear em escala
  log com `expm1`. Em alguns eventos, um log previsto alto vira um número astronômico de demitidos.
  Ela serve como referência (baseline), não como candidata.

Artefatos: `modelo_tech_layoffs/modelo.joblib` (MLP + pré-processamento + log1p/expm1 + corte em 0),
`metricas.json` e `amostra_conferencia.json` (6 entradas do teste; a última tem `Stage` e
`Money_Raised_clean` nulos).

## 6. Análise do erro (MLP, conjunto de teste)

| Medida | Valor |
|---|---|
| Erro absoluto mediano | 45,1 |
| Percentil 90 do erro absoluto | 389,6 |
| Maior erro absoluto | 7.936,3 |
| MAE nos 18 eventos com mais de 1.000 demitidos | 2.171,4 |
| MAE nos demais 331 eventos | 130,8 |
| Previsões negativas antes do corte | 0 (menor previsão bruta: 10,917) |
| Previsões acima de `Company_Size_before_Layoffs` | **14 de 349** |

- O MAE de 236,03 é dominado pelos poucos eventos grandes. Para um evento típico, o erro é bem menor
  que o MAE.
- **Inconsistência registrada, não corrigida:** em 14 eventos de teste (todos de empresas com até 120
  funcionários), o modelo estima mais demitidos do que funcionários existentes. Nada foi alterado na
  avaliação para esconder isso. A função `prever_demissoes()` emite um alerta nesses casos.

## 7. Limitações

- O dataset só tem **eventos registrados** de demissão; empresas que não demitiram não aparecem.
- O modelo estima **quantas pessoas** foram demitidas num evento com aquelas características. Ele
  **não** estima a probabilidade de uma empresa demitir.
- A divisão é **aleatória**: ela não mostra desempenho em anos futuros.
- **Empresas repetidas:** 121 das 327 empresas do teste também aparecem no treino (131 eventos de
  teste). Isso pode deixar as métricas otimistas.
- **Categorias raras:** 30 setores e 25 países têm menos de 5 eventos. No teste há categorias que
  não existem no treino: país `South Korea` e setores `Appliances`, `Financial Services`, `Logistic`,
  `Motor Vehicle Manufacturing`, `Semiconductor Manufacturing` e
  `The Animation Tool for UI/UX Designers.`. O `OneHotEncoder` ignora essas categorias.
- **Alvo muito assimétrico:** assimetria 9,55; mediana 90; máximo 22.000.
- **`Year` × `Date_layoffs`:** 7 linhas divergem. `year` vem de `Year`, como no notebook.
- **Capital captado:** a unidade e a moeda de `Money_Raised_in__mil` (milhões, provavelmente US$)
  não estão confirmadas.
- **Porte anterior:** `Company_Size_before_Layoffs` é a entrada mais associada ao alvo (Spearman
  0,798). Num cenário real, ele precisa ser conhecido antes do anúncio.
- **MAE não é intervalo de confiança:** "previsão ± MAE" mostra o erro médio típico, não uma faixa
  com probabilidade garantida.
- **R² não é acurácia:** um R² de 0,5537 significa que o modelo explica cerca de 55% da variância
  do alvo no teste. Não significa "acerta 55% das vezes".
- Paralelismo: `n_jobs=1` no GridSearch, mas o MLP usa as threads da biblioteca numérica (cerca de
  400% de CPU observados), sem efeito nos resultados.
