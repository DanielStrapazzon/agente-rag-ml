# Etapa 01 — Organização e integração inicial (Tech Layoffs + Agente RAG)

> Escopo desta etapa: **analisar, organizar e incorporar** os arquivos do trabalho anterior.
> Fora do escopo: integração definitiva do RAG, retreino/exportação de modelos, troca do domínio
> em `ferramentas.py`, substituição de `dados/dataset_legivel.csv` ou da base `conhecimento/`.

Data da análise: 2026-11-26.

---

## 1. Estrutura atual do projeto (antes desta etapa)

Arquivos rastreados no commit `dec6d9f` ("Kit inicial do agente + enunciado"):

```text
agente-rag-ml/
├── app.py                         servidor Flask (página + /api/chat + /saude)
├── agente.py                      pipeline interpretar → recuperar → prever → gerar → verificar
├── recuperacao.py                 RAG: TF-IDF (1-2 gramas) + cosseno sobre conhecimento/*.md + fatos do CSV
├── ferramentas.py                 dados/modelo: preparar, para_o_modelo, prever, casos semelhantes, estatísticas, fatos
├── treinar_exemplo.py             gera dataset SINTÉTICO de aluguel + treina KNN (exemplo do kit)
├── conferir.py                    "gabarito": estatísticas e previsões a partir do CSV e do modelo
├── testar_sem_api.py              teste offline (sem Gemini) de dados, ponte, modelo, RAG
├── comparar_sem_rag.py            mesmas perguntas com e sem RAG (usa a API)
├── conhecimento/analise_exploratoria.md   base do RAG (EXEMPLO de aluguel)
├── prompts/interpretar.md         prompt da etapa 1 (JSON)
├── prompts/responder.md           persona "Assistente de Aluguéis"
├── dados/dataset_legivel.csv      1.500 linhas sintéticas de aluguel (area_m2, quartos, vagas, bairro, aluguel)
├── modelo/modelo.joblib           Pipeline KNN do EXEMPLO do kit
├── modelo/metricas.json           KNN k=7, R² 0,944, MAE 247,5, sklearn 1.9.1 (EXEMPLO)
├── modelo/amostra_conferencia.json   5 entradas de aluguel (EXEMPLO)
├── static/index.html              página de chat (textos de aluguel)
├── exemplo_classificacao/         variante de classificação (crédito)
├── exemplo_transformacao/         variante com transformações fora do Pipeline
├── versao_streamlit/              mesma interface em Streamlit
├── requirements.txt, links.txt, .env.example, .gitignore, .python-version (3.13), README.md
```

Estado do Git no início da etapa (não commitado):

| Item | Situação |
|---|---|
| `dados/tech_layoffs.csv` | **já adicionado ao índice** (staged) pelo usuário; hash idêntico ao original do `dist/` |
| `notebooks/analise_modelos.ipynb` | **já adicionado ao índice** (staged); hash idêntico ao original do `dist/` |
| `.gitignore` | modificado: linha `dist` acrescentada (sem quebra de linha final) |
| `requirements.txt` | modificado **apenas em comentário**: "no notebook/Colab" → "no notebooks/Colab" (provável renomeação automática da IDE; versões inalteradas) |
| `.idea/` | não rastreado (configuração do PyCharm) |
| `dist/` | **não rastreado** (`git ls-files dist` vazio) e já ignorado pela linha `dist` |

Ambiente: só há Python 3.14 do sistema, sem venv e sem pandas/sklearn. O kit exige 3.12/3.13
(numpy 2.5.3, scikit-learn 1.9.1).

## 2. Estrutura identificada em `dist/`

O ZIP extraído tem uma subpasta adicional:

```text
dist/Trabalho_Tech_Layoffs_DADOS_REAIS-20261126T234301Z-1-001/
└── Trabalho_Tech_Layoffs_DADOS_REAIS/
    ├── README.md                        instruções do trabalho anterior (Colab), alvo e cuidados
    ├── requirements.txt                 dependências do notebook (faixas >=)
    ├── data/tech_layoffs.csv            dataset real (1.745 linhas × 18 colunas)
    ├── notebooks/analise_modelos.ipynb  EDA + regressão (Linear, KNN, MLP) com saídas executadas
    ├── referencias/notebook_EDA_kaggle.ipynb   notebook EDA de terceiros (Kaggle), referência
    └── reports/
        ├── ROTEIRO_RELATORIO.md         roteiro do relatório técnico
        └── resultados/                  12 PNG + 2 CSV gerados pelo notebook
            ├── heatmap.png, boxplot.png, histograma.png
            ├── frequencia_Country.png, frequencia_Industry.png, frequencia_Year.png
            ├── previsto_real_{KNN,MLP,Regressao_Linear_Multipla}.png
            ├── residuos_{KNN,MLP,Regressao_Linear_Multipla}.png
            ├── metricas_teste.csv
            └── melhores_parametros.csv
```

## 3. Responsabilidade de cada arquivo do trabalho anterior

| Arquivo | Responsabilidade |
|---|---|
| `data/tech_layoffs.csv` | fonte original dos dados (não transformada) |
| `notebooks/analise_modelos.ipynb` | carga/validação, EDA (Seaborn), pipeline de pré-processamento, GridSearchCV de 3 regressores, métricas, gráficos, limitações |
| `reports/resultados/*.png` | figuras da EDA e da avaliação dos modelos |
| `reports/resultados/metricas_teste.csv` | R², RMSE, MAE no teste dos 3 modelos |
| `reports/resultados/melhores_parametros.csv` | grades testadas, melhores parâmetros e MAE médio de CV |
| `reports/ROTEIRO_RELATORIO.md` | roteiro do relatório acadêmico |
| `referencias/notebook_EDA_kaggle.ipynb` | EDA de referência do Kaggle (lê `/kaggle/input/.../tech_layoffs.xlsx`; não executável localmente sem ajustes; usa plotly) |
| `README.md` | instruções Colab, requisitos acadêmicos, fonte, interpretação do alvo |
| `requirements.txt` | dependências do notebook |

## 4. Mapeamento origem → destino

Origem relativa a `dist/Trabalho_Tech_Layoffs_DADOS_REAIS-20261126T234301Z-1-001/Trabalho_Tech_Layoffs_DADOS_REAIS/`.

| Origem | Destino | Ação |
|---|---|---|
| `data/tech_layoffs.csv` | `dados/tech_layoffs.csv` | já presente (staged); conferido por hash, sem cópia nova |
| `notebooks/analise_modelos.ipynb` | `notebooks/analise_modelos.ipynb` | já presente (staged); conferido por hash |
| `reports/resultados/` (14 arquivos) | `reports/resultados/` | cópia |
| `reports/ROTEIRO_RELATORIO.md` | `reports/ROTEIRO_RELATORIO.md` | cópia |
| `referencias/notebook_EDA_kaggle.ipynb` | `referencias/notebook_EDA_kaggle.ipynb` | cópia |
| `README.md` | `reports/README_TRABALHO_ANTERIOR.md` | cópia renomeada |
| `requirements.txt` | `reports/requirements_trabalho_anterior.txt` | cópia renomeada |

Cópias feitas com `cp -p` (preserva conteúdo e datas); `dist/` permanece intacto.

## 5. Dataset `dados/tech_layoffs.csv`

- **1.745 registros, 18 colunas**, UTF-8, separador vírgula, sem duplicatas completas.
- Datas `Date_layoffs` em ISO `AAAA-MM-DD`, todas válidas, de **2020-03-12 a 2025-12-28**.
- `Year`: 2020 (330), 2021 (14), 2022 (568), 2023 (474), 2024 (265), 2025 (94).
  Em 7 linhas `Year` difere do ano de `Date_layoffs`.

| Coluna | Tipo | Ausentes | Observações |
|---|---|---|---|
| `Nr` | inteiro (id) | 0 | 1–2412, não contínuo; excluído do modelo |
| `Company` | categórico | 0 | 1.327 empresas; 298 aparecem em mais de um evento |
| `Location_HQ` | categórico | 0 | 181 cidades |
| `Region` | categórico | 0 | 4 valores (`other` = 1.168) |
| `USState` | categórico | 1 | 66 valores; usa `non` como marcador fora dos EUA |
| `Country` | categórico | 0 | 42 países (USA = 1.121) — **entrada do modelo** |
| `Continent` | categórico | 0 | 6 valores (grafia `Oceana`) — **entrada do modelo** |
| `Laid_Off` | numérico | 0 | **alvo**; 3 a 22.000, mediana 90, muito assimétrico |
| `Date_layoffs` | data (texto) | 0 | origem de `month` |
| `Percentage` | numérico | 0 | 0–100; **excluído (vazamento)** |
| `Company_Size_before_Layoffs` | numérico | 0 | 4 a 600.000 — **entrada do modelo** |
| `Company_Size_after_layoffs` | numérico | 0 | **excluído (vazamento)** |
| `Industry` | categórico | 0 | 60 valores — **entrada do modelo** |
| `Stage` | categórico | 27 | 16 valores, inclui `Unknown` (228) — **entrada do modelo** |
| `Money_Raised_in__mil` | numérico | 110 | 1 a 121.900 (US$ mi) — **entrada do modelo** (como `Money_Raised_clean`) |
| `Year` | inteiro | 0 | **entrada do modelo** (como `year`) |
| `latitude`, `longitude` | numérico | 0 | não usados no modelo |

Atributos derivados no notebook: `Money_Raised_clean` (numérico), `month` (de `Date_layoffs`),
`year` (de `Year`).

`dados/dataset_legivel.csv` continua sendo o **exemplo sintético de aluguel** do kit e **não foi
alterado** nesta etapa (ferramentas.py, modelo e amostra de conferência dependem dele).

## 6. Análise do notebook `analise_modelos.ipynb`

**Natureza:** o notebook **treina modelos de regressão** capazes de produzir estimativas — não é só
análise estatística. Estima `Laid_Off` (quantidade de demitidos) **em eventos de demissão já
registrados**; **não** prevê se haverá demissões no futuro. A divisão é aleatória, então também não
valida previsão temporal (anos futuros).

| Item | Conteúdo |
|---|---|
| Limpeza | `drop_duplicates`; `Laid_Off` numérico e ≥ 0; conversões numéricas com `errors='coerce'`; `Date_layoffs` → datetime |
| EDA | heatmap de correlação, boxplot e histograma do alvo, frequências (top 12) de Industry, Country, Year; mediana do alvo = 90 |
| Alvo | `Laid_Off` |
| Entradas (8) | categóricas `Country`, `Continent`, `Industry`, `Stage`; numéricas `Company_Size_before_Layoffs`, `Money_Raised_clean`, `year`, `month` |
| Excluídos | `Percentage`, `Company_Size_after_layoffs` (vazamento), `Nr` |
| Pré-processamento | `ColumnTransformer`: numéricas = `SimpleImputer(median)` + `StandardScaler`; categóricas = `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`; **dentro do Pipeline** (caso 1 do kit) |
| Alvo transformado | `TransformedTargetRegressor(log1p / expm1)` |
| Divisão | `train_test_split` 80/20, `random_state=42` → 1.396 treino / 349 teste |
| Validação | `GridSearchCV`, `KFold(5, shuffle, seed 42)`, `scoring='neg_mean_absolute_error'` |
| Algoritmos e grades | Regressão Linear (sem grade); KNN: `n_neighbors` [3,7,15], `weights` [uniform, distance], `metric` [euclidean, manhattan]; MLP (`max_iter=400`, `early_stopping`, `n_iter_no_change=15`, seed 42): `hidden_layer_sizes` [(32,), (64,32)], `activation` [relu, tanh], `alpha` [0.001, 0.1] |
| Pós-processamento | previsões truncadas em 0 (`np.maximum(pred, 0)`) |
| Melhores parâmetros | KNN: euclidean, k=7, distance · MLP: tanh, alpha 0.1, (64, 32) |

Métricas no teste (de `reports/resultados/metricas_teste.csv`, valores registrados pelo notebook):

| Modelo | R² | RMSE | MAE | MAE médio CV |
|---|---|---|---|---|
| **MLP** | 0,5537 | 851,53 | **236,03** | 201,19 |
| KNN | 0,5186 | 884,34 | 258,29 | 210,19 |
| Regressão Linear Múltipla | −41,73 | 8.331,71 | 884,24 | 101.025,85 |

Melhor modelo pelo menor MAE de teste: **MLP**. O próprio notebook pede que R², RMSE, outliers e
resíduos também sejam considerados antes de concluir.

**Limitações registradas no notebook:** só estima eventos já registrados; disponibilidade de
`Company_Size_before_Layoffs` antes do evento precisa ser justificada; `year`/`month` pressupõem um
período definido; partição aleatória não testa anos futuros; empresas repetidas em vários eventos
(possível vazamento entre treino e teste).

**Observações adicionais desta análise (não alteram a metodologia):**
- A EDA limpa `Money_Raised_in__mil` com regex, o pipeline usa `pd.to_numeric` direto. Neste CSV a
  coluna já é numérica, então o resultado é o mesmo — mas a diferença deve ser lembrada se o CSV mudar.
- A regressão linear explode na escala original (R² negativo) por causa do `expm1` sobre o log.
- O notebook **não salva** nenhum modelo (`joblib.dump` ausente) nem registra versões de sklearn/numpy.
- O README anterior cita a página Kaggle "tech-layoffs-2020-2024", mas o CSV vai até 2025-12;
  a versão exata da fonte precisa ser confirmada.

**Dependência de Colab / caminhos:** a célula 1 tenta `google.colab.drive.mount` e usa
`ROOT = /content/drive/MyDrive/Trabalho_Tech_Layoffs_DADOS_REAIS`; fora do Colab cai no `except
ImportError` e usa a pasta pai de `notebooks/`. Ela exige **`ROOT/data/tech_layoffs.csv`**, mas no
repositório o arquivo está em **`dados/`**. Para executar localmente será preciso (próxima etapa,
com autorização): trocar `'data'` por `'dados'` na célula 1 e 3 (ou aceitar ambos). As saídas
continuariam em `reports/resultados/`, sobrescrevendo as figuras originais — recomenda-se
redirecionar para outra pasta numa nova execução para preservar os resultados acadêmicos.

## 7. Artefatos de ML (`modelo/`)

Confirmado: `modelo/modelo.joblib`, `metricas.json` e `amostra_conferencia.json` pertencem ao
**exemplo do kit** (gerados por `treinar_exemplo.py`; KNN de aluguel, R² 0,944, MAE 247,5). Foram
preservados sem alteração. Não há modelo exportado do trabalho anterior.

## 8. RAG atual e adaptação futura

Como funciona hoje:
1. `recuperacao.Recuperador` lê todos os `conhecimento/*.md`, quebra em parágrafos/itens de lista
   (prefixados pelo título da seção) e soma as frases de `ferramentas.fatos_do_dataset()`.
2. TF-IDF (`strip_accents`, stopwords PT, 1-2 gramas) + cosseno; `buscar(k=4, limiar=0.05)`.
3. `agente.py` usa o campo `busca` (pergunta reescrita pelo Gemini) para buscar; junta estatísticas
   (`F.estatisticas`) e, se for previsão, `prever` + `casos_semelhantes` no CONTEXTO.
4. Etapa 5 confere se todo número da resposta está no CONTEXTO.

Adaptações necessárias (próximas etapas):
- `ferramentas.py`: `ALVO="Laid_Off"`, `NUMERICAS`/`CATEGORICAS` do notebook, `UNIDADE_ALVO`
  ("funcionários demitidos"), `ATRIBUTOS` com descrições, `_regras_do_dominio()` (ex.: porte > 0,
  mês 1–12), `fatos_do_dataset()` e `para_o_modelo()` (gerar `month`/`year` a partir de data, se for
  o caso). Muitos níveis categóricos (60 indústrias, 42 países) → `casos_semelhantes()` agrupa pela
  1ª categórica; avaliar.
- `dados/dataset_legivel.csv`: gerar a partir de `tech_layoffs.csv` só com a limpeza do notebook.
- Termos da base estão em inglês (colunas/categorias) e as perguntas em português: o TF-IDF não
  casa "indústria"↔"Industry"; a base deve mencionar os dois.
- `prompts/responder.md`, `static/index.html`, `versao_streamlit/streamlit_app.py`,
  `comparar_sem_rag.py`: trocar textos de aluguel.

Estrutura proposta para a base (apenas proposta; nada criado):

```text
conhecimento/
├── descricao_dataset.md       colunas, tipos, ausentes, período, fonte, o que é cada evento
├── analise_exploratoria.md    achados da EDA (a partir dos gráficos reais)
├── modelo_metricas.md         pipeline, grades, melhores parâmetros, métricas reais
└── limitacoes.md              vazamento, divisão aleatória, eventos já registrados, empresas repetidas
```

## 9. Dependências

| Pacote | Kit (`requirements.txt`) | Trabalho anterior | Uso |
|---|---|---|---|
| pandas | ==3.0.5 | >=2.2 | compartilhada (produção + notebook) |
| numpy | ==2.5.3 | >=1.26 | compartilhada |
| scikit-learn | ==1.9.1 | >=1.4 | compartilhada — **versão de treino deve ser igual à de produção** |
| joblib | ==1.6.0 | — (vem com sklearn) | produção |
| flask, gunicorn, google-genai, python-dotenv | fixas | — | só produção (kit) |
| streamlit | ==1.67.0 (`versao_streamlit/`) | — | só versão Streamlit |
| matplotlib | — | >=3.8 | só notebook |
| seaborn | — | >=0.13 | só notebook |
| notebook | — | >=7 | só notebook |
| plotly | — | — (não listado) | usado só por `referencias/notebook_EDA_kaggle.ipynb` |

Conflitos: não há conflito direto — as faixas `>=` do trabalho anterior aceitam as versões fixas do
kit. Risco: o notebook foi executado no Colab com versões desconhecidas; o modelo final deve ser
**retreinado no venv do kit** (sklearn 1.9.1 / numpy 2.5.3), não importado do Colab.
Decisão: `requirements.txt` do kit **mantido**; dependências de notebook não entram em produção.

## 10. Possíveis conflitos

- `dados/` vs `data/`: notebook espera `data/` (ver §6).
- Re-executar o notebook sobrescreve `reports/resultados/` originais.
- `dataset_legivel.csv` / `modelo/` / `conhecimento/` ainda são de aluguel: o app continua
  respondendo sobre aluguel até a adaptação.
- `.gitignore` original tinha `dist` sem âncora — ignoraria qualquer arquivo/pasta chamado `dist`
  em qualquer nível; trocado por `/dist/`.
- `requirements.txt` com alteração acidental de comentário (não funcional) pendente de decisão do usuário.

## 11. Decisões técnicas e justificativas

1. **Copiar, não mover**, de `dist/` → preserva a origem; `dist/` continua ignorado.
2. **Não reaplicar** CSV e notebook já staged — hashes idênticos aos originais.
3. **README e requirements anteriores renomeados** dentro de `reports/` → não sobrescrevem os do kit.
4. **Não tocar** em `dataset_legivel.csv`, `modelo/`, `conhecimento/`, prompts e código → kit segue
   funcionando "de fábrica" até a adaptação coordenada.
5. **Não editar o notebook** (nem o caminho `data/`) → metodologia e saídas preservadas; a mudança
   fica documentada como pendência.
6. **`.gitignore`**: `/dist/` (âncora na raiz), `.ipynb_checkpoints/`, `.pytest_cache/`, `.idea/`
   (configuração local da IDE). Não ignorar `.csv`, `.ipynb`, `.joblib`.
7. **Não instalar pacotes** nem criar venv nesta etapa.

## 12. Checklist

- [x] Analisar estrutura atual do repositório
- [x] Analisar recursivamente `dist/` (subpasta extra identificada)
- [x] Ler README do kit e do trabalho anterior
- [x] Analisar `analise_modelos.ipynb` (células e saídas)
- [x] Perfilar `tech_layoffs.csv` (colunas, tipos, ausentes, datas)
- [x] Analisar RAG (`recuperacao.py`, `ferramentas.py`, `agente.py`, prompts, conhecimento)
- [x] Comparar dependências
- [x] Confirmar que `modelo/` é do exemplo do kit
- [x] Verificar se `dist/` está rastreado (não está)
- [x] Criar `plan/01-organizacao-projeto.md`
- [x] Conferir por hash `dados/tech_layoffs.csv` e `notebooks/analise_modelos.ipynb` já presentes
- [x] Copiar `reports/resultados/` (14 arquivos)
- [x] Copiar `reports/ROTEIRO_RELATORIO.md`
- [x] Copiar `referencias/notebook_EDA_kaggle.ipynb`
- [x] Copiar README → `reports/README_TRABALHO_ANTERIOR.md`
- [x] Copiar requirements → `reports/requirements_trabalho_anterior.txt`
- [x] Atualizar `.gitignore`
- [x] Acrescentar seção Tech Layoffs + "O que mudamos em relação ao kit" no README
- [x] Validar cópias por SHA-256
- [x] Verificar duplicatas fora de `dist/` e que nenhum arquivo funcional do kit mudou
- [ ] Executar `testar_sem_api.py` — **não executado**: não há Python 3.12/3.13 nem dependências instaladas (só Python 3.14 do sistema)
- [x] Registrar `git status` final

## 13. Pendências para as próximas etapas

1. Criar venv Python 3.13 e instalar `requirements.txt`; rodar `testar_sem_api.py` e `conferir.py`
   (linha de base do kit).
2. Decidir sobre a alteração de comentário em `requirements.txt` (manter ou reverter).
3. Ajustar caminho `data/` → `dados/` no notebook (ou script de treino) e redirecionar saídas.
4. Criar `treinar.py` a partir do notebook (mesmo pipeline/hiperparâmetros, modelo escolhido
   justificado — MLP pelo MAE ou KNN), rodando no venv do kit.
5. **Exportar o modelo** treinado (`best_estimator_`) em `modelo/modelo.joblib`.
6. **Gerar `metricas.json` reais** (`r2_teste`, `mae_teste`, `sklearn_version`, `modelo`).
7. **Gerar `amostra_conferencia.json`** com entradas legíveis do Tech Layoffs.
8. **Adaptar `para_o_modelo()`** (colunas derivadas `year`, `month`, `Money_Raised_clean`).
9. **Validar compatibilidade** treino ↔ aplicação (`conferir_transformacao()` → `ok: True`).
10. Gerar `dados/dataset_legivel.csv` do Tech Layoffs (substituindo o de aluguel, junto com 5–9).
11. Reescrever `conhecimento/` na estrutura proposta, com base nos resultados reais.
12. Adaptar prompts, página, Streamlit e `comparar_sem_rag.py` ao domínio.
13. Confirmar a versão exata do dataset no Kaggle (o CSV vai até 2025-12).
14. Preencher `links.txt` e seguir o roteiro do relatório.
