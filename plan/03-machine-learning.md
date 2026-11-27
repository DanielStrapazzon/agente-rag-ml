# Etapa 03 — Treinamento e exportação do Machine Learning Tech Layoffs

Data: 2026-11-26. Base: commit `ca842e9` (etapas 01 e 02 versionadas).
Implementação feita com apoio do **Claude Code** (Anthropic), revisada e executada pelo grupo.

## 1. Objetivos

1. `treinar.py`: reproduzir o treino do notebook (3 algoritmos, mesmas grades, mesma divisão)
   reaproveitando `preparar_dados.preparar()` e `separar_atributos_modelo()`.
2. Selecionar o modelo pelo **menor MAE de validação cruzada** e exportar para
   `modelo_tech_layoffs/` (`modelo.joblib`, `metricas.json`, `amostra_conferencia.json`).
3. Função de previsão reutilizável, independente do Gemini (`previsao_layoffs.py`).
4. Comparação com os resultados históricos em `reports/novo_treinamento/`.
5. Testes em `tests/test_machine_learning.py` usando os artefatos gerados (sem refazer o GridSearch).

Fora do escopo: alterar `ferramentas.py`, `agente.py`, `app.py`, `recuperacao.py`, prompts, frontend,
`conhecimento/`, `modelo/`, `dados/dataset_legivel.csv`, `reports/resultados/`.

## 2. Metodologia anterior (confirmada nas células 7 e 9 do notebook)

| Item | Notebook |
|---|---|
| Dados | `drop_duplicates`, alvo numérico ≥ 0 (preparado em `preparar_dados.py`) |
| Entradas | cat `Country, Continent, Industry, Stage`; num `Company_Size_before_Layoffs, Money_Raised_clean, year, month` |
| Alvo | `Laid_Off` (float64) |
| Divisão | `train_test_split(test_size=0.2, random_state=42)` |
| Pré-processamento | `ColumnTransformer`: num = `SimpleImputer(median)` + `StandardScaler`; cat = `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` |
| Estimador | `TransformedTargetRegressor(regressor=Pipeline([('prep', prep), ('model', modelo)]), func=np.log1p, inverse_func=np.expm1)` |
| Busca | `GridSearchCV(cv=KFold(5, shuffle=True, random_state=42), scoring='neg_mean_absolute_error', n_jobs=1, refit=True, error_score='raise')`; Regressão Linear com grade `[{}]` |
| KNN | `n_neighbors` [3, 7, 15], `weights` [uniform, distance], `metric` [euclidean, manhattan] |
| MLP | `MLPRegressor(random_state=42, max_iter=400, early_stopping=True, n_iter_no_change=15)`; `hidden_layer_sizes` [(32,), (64, 32)], `activation` [relu, tanh], `alpha` [0.001, 0.1] |
| Pós-processamento | `np.maximum(gs.predict(X_test), 0)` — **só nas métricas de teste**; o MAE de CV usa a previsão sem corte |
| Métricas | R², RMSE = `sqrt(mean_squared_error)`, MAE (teste); MAE médio CV = `-best_score_` |
| Escolha no notebook | "menor MAE no conjunto de teste" (célula 11) |

## 3. Arquitetura do treinamento

```text
preparar_dados.preparar()              (limpeza única, etapa 02)
        │
separar_atributos_modelo() → X (8 col.), y
        │   + conferência: X/y do CSV legível == X/y da preparação
train_test_split 80/20, seed 42
        │
para cada algoritmo: GridSearchCV( TransformedTargetRegressor( Pipeline(prep, modelo) ) )
        │   MAE CV (sem corte, como no notebook)
        │   previsão de teste com corte em zero → R², RMSE, MAE
seleção: menor MAE de CV
        │
EstimadorNaoNegativo(best_estimator_)   (corte em zero dentro do objeto salvo)
        │
modelo_tech_layoffs/modelo.joblib + metricas.json + amostra_conferencia.json
reports/novo_treinamento/metricas_teste.csv + melhores_parametros.csv
```

## 4. Atributos

Os oito do notebook, mesma ordem, nomes e tipos (categóricas `object` com `np.nan`, numéricas
`float64`). Excluídas por vazamento: `Percentage`, `Company_Size_after_layoffs`. Sem normalização
de grafias. `year` vem de `Year` (7 linhas divergem de `Date_layoffs`; ver `reports/PERFIL_DATASET.md`).

## 5. Critério de seleção

**Menor MAE médio da validação cruzada (5 folds, só treino).** O conjunto de teste é usado apenas
para relatar desempenho. Diferença em relação ao notebook: lá a escolha declarada foi pelo menor MAE
de teste; aqui o teste não participa da escolha, para evitar seleção retrospectiva. Se os dois critérios
apontarem modelos diferentes, isso será registrado.

## 6. Estratégia de avaliação

- Por algoritmo: MAE CV, MAE/RMSE/R² de teste, melhores parâmetros, tempo e avisos de convergência.
- Comparação com `reports/resultados/metricas_teste.csv` e `melhores_parametros.csv` (históricos).
- Contagem de previsões de teste maiores que `Company_Size_before_Layoffs` (inconsistência registrada,
  não corrigida).

## 7. Formato dos artefatos

- `modelo.joblib`: `previsao_layoffs.EstimadorNaoNegativo` envolvendo o `best_estimator_`
  (`TransformedTargetRegressor` → `Pipeline(prep, modelo)`). `predict()` = `max(expm1(...), 0)`.
  A classe fica num módulo importável (não em `__main__`); o carregamento exige `previsao_layoffs.py`
  e `preparar_dados.py` na raiz do projeto (como `ferramentas.py`).
- `metricas.json`: contrato do kit (`modelo`, `r2_teste`, `mae_teste`, `sklearn_version`) + campos
  pedidos (tarefa, rmse, mae_cv, parâmetros, versões, seed, tamanhos, hashes, comparação).
- `amostra_conferencia.json`: mesmo formato do kit (`[{"entrada": {...}, "previsao_no_treino": x}]`),
  entradas legíveis do teste; ausentes como `null` → convertidos para `np.nan` por
  `previsao_layoffs.para_dataframe_modelo()` e imputados pelo Pipeline.

## 8. Arquivos envolvidos

| Arquivo | Ação |
|---|---|
| `previsao_layoffs.py` | criar (pós-processamento, wrapper serializável, conversão de entrada, previsão) |
| `treinar.py` | criar |
| `modelo_tech_layoffs/*` | gerar |
| `reports/novo_treinamento/*` | gerar / escrever |
| `tests/test_machine_learning.py` | criar |
| `plan/03-machine-learning.md`, `README.md` | criar / atualizar |
| `modelo/*`, `reports/resultados/*`, `dados/*.csv`, código do kit | **não alterar** |

## 9. Checklist

- [x] Ler planos, notebook, preparação, kit e contratos
- [x] Confirmar versões do ambiente
- [x] Criar este plano
- [x] Implementar `previsao_layoffs.py`
- [x] Implementar `treinar.py`
- [x] Executar o treino e registrar avisos
- [x] Validar carregamento em novo processo e reprodução da amostra
- [x] Escrever `reports/novo_treinamento/COMPARACAO_TREINAMENTO.md`
- [x] Implementar e executar `tests/test_machine_learning.py`
- [x] Reexecutar `testar_sem_api.py` e `conferir.py`; conferir hashes protegidos
- [x] Atualizar README e este plano com os resultados

## 10. Estratégia de testes

Usam os artefatos já gerados (sem GridSearch): seleção das 8 entradas e ausência de vazamento; tipos;
carregamento do `.joblib`; presença de `ColumnTransformer`/imputadores/`TransformedTargetRegressor`;
previsões válidas, com ausentes, com categoria desconhecida; corte de negativos; ausência de
NaN/Inf; reprodução da amostra (tolerância relativa 1e-9); campos do `metricas.json` compatíveis com
`ferramentas.py`; JSON estrito; hash do CSV original e dos arquivos de `modelo/`.

## 11. Decisões técnicas

1. **Corte em zero serializado** via `EstimadorNaoNegativo` (em vez de mudar `inverse_func`): preserva o
   GridSearch idêntico ao notebook (CV sem corte) e garante que avaliação, amostra, conferência do kit e
   agente usem a mesma previsão.
2. `n_jobs=1` (como no notebook): sem paralelismo, memória previsível e resultados determinísticos.
3. Métricas exportadas sem arredondamento excessivo (R² 4 casas, MAE/RMSE 2 casas no contrato do kit;
   valores completos nos CSVs).
4. Avisos (`ConvergenceWarning` etc.) capturados e contados por algoritmo.

## 12. Resultados

Ambiente do treino: Python 3.14.10, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5, joblib 1.6.0.
`.venv/bin/python treinar.py`: ~24 s, **nenhum aviso** (nem `ConvergenceWarning`); duas execuções
geraram `metricas.json`, `amostra_conferencia.json` e os CSVs **idênticos** (SHA-256).
CSV legível equivalente à preparação: `True`. Carregamento em novo processo: amostra reproduzida, 0 divergências.

| Modelo | MAE CV | MAE teste | RMSE | R² | Melhores parâmetros |
|---|---:|---:|---:|---:|---|
| **MLP** | **201,19** | **236,03** | **851,53** | **0,5537** | tanh, alpha 0.1, (64, 32) |
| KNN | 210,19 | 258,29 | 884,34 | 0,5186 | euclidean, k=7, distance |
| Regressão Linear | 101.025,85 | 884,24 | 8.331,71 | −41,7277 | padrão |

- **Comparação com o histórico:** idêntico (diferenças ≤ 1,1 × 10⁻¹³; mesmos parâmetros e grades).
- **Selecionado:** MLP pelo menor MAE de CV; o menor MAE de teste também é do MLP (critérios concordam).
- Previsões brutas negativas no teste: 0 (o corte em zero não alterou nenhuma métrica).
- Previsões acima de `Company_Size_before_Layoffs`: 14 de 349 (todas em empresas com até 120 funcionários) — registrado, não corrigido.
- Erro: mediano 45,1; MAE 2.171,4 nos 18 eventos > 1.000 contra 130,8 nos demais.
- Empresas do teste também presentes no treino: 121 de 327 (131 eventos).
- Categorias do teste ausentes do treino: `South Korea`; 6 setores (ver comparação).

Testes: `.venv/bin/python -m unittest discover -s tests -v` → **41 testes OK** (20 da etapa 02 + 21 do modelo).
Kit: `testar_sem_api.py` e `conferir.py` sem mudanças (ponte `ok: True`, previsão 4199,5); hashes de
`modelo/*`, `dados/*.csv` inalterados; `git diff HEAD` vazio para arquivos rastreados.

Documentação: `reports/novo_treinamento/COMPARACAO_TREINAMENTO.md`, README (seção Tech Layoffs).

## 13. Limitações (detalhadas na comparação)

Eventos só registrados (sem empresas que não demitiram); não é probabilidade de evento; divisão
aleatória (não temporal); empresas repetidas entre treino e teste; categorias raras e não vistas;
alvo muito assimétrico; 7 linhas `Year` ≠ data; unidade/moeda do capital não confirmada;
disponibilidade do porte anterior; MAE ≠ intervalo de confiança; R² ≠ acurácia; vantagem MLP × KNN
pequena com uma só divisão; MLP usa várias threads de BLAS (~400% CPU) apesar de `n_jobs=1`.

## 14. Pendências para a integração (etapa 04)

1. Copiar `modelo_tech_layoffs/*` para `modelo/` **junto** com a troca do dataset e de `ferramentas.py`
   (o kit lê `modelo/` e `dados/dataset_legivel.csv` fixos).
2. `ferramentas.py`: `ALVO`, `NUMERICAS`, `CATEGORICAS`, `UNIDADE_ALVO`, `ATRIBUTOS` (descrições em
   português), `_regras_do_dominio()` (porte > 0, mês 1–12, ano plausível), `fatos_do_dataset()`.
3. `para_o_modelo()` → usar `previsao_layoffs.para_dataframe_modelo()` (tipos e `null` → `np.nan`);
   a conferência do kit compara `MODELO.predict(para_o_modelo(...))` com a amostra, que já inclui o corte em zero.
4. `preparar()` do kit trata categoria desconhecida como **erro**; decidir se mantém (recomendado,
   com as opções válidas) ou se aceita com alerta. Ausentes: o kit pede o dado que falta; avaliar se
   `Stage`/`Money_Raised_clean` podem ser opcionais (o Pipeline imputa).
5. `casos_semelhantes()` filtra pela 1ª categórica (`Country`): com 42 países e USA = 64% dos eventos,
   avaliar outro critério.
6. Alertar quando a previsão supera o porte (já em `prever_demissoes()`), e informar o MAE com a
   ressalva de que não é intervalo de confiança.
7. Escrever `conhecimento/` (descrição, EDA, modelo/métricas, limitações) a partir de
   `PERFIL_DATASET.md` e `COMPARACAO_TREINAMENTO.md`.
8. Hospedagem: confirmar Python 3.14.10 no Render/Streamlit (`PYTHON_VERSION`); `.python-version` diz 3.13.
