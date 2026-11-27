# Etapa 02 — Preparação do ambiente e do dataset Tech Layoffs

Data: 2026-11-26. Base: commit `e8a2beb` (etapa 01 versionada).

## 1. Contextualização

A etapa 01 organizou no repositório os arquivos do trabalho anterior (dataset, notebook, resultados).
Esta etapa cria um processo **reproduzível** de preparação do `dados/tech_layoffs.csv`, extraído do
notebook `notebooks/analise_modelos.ipynb`, gerando um dataset **legível** (sem transformações de
modelo) que servirá às estatísticas, ao RAG, aos casos semelhantes e ao treino da etapa 03.

Fora do escopo: treino/exportação de modelo, alterações em `agente.py`, `app.py`, `ferramentas.py`,
`recuperacao.py`, prompts, frontend, `conhecimento/`, `modelo/` e `dados/dataset_legivel.csv`.

## 2. Situação do ambiente (confirmada por código)

| Item | Valor |
|---|---|
| Interpretador | `.venv/bin/python` → **Python 3.14.10** |
| scikit-learn / numpy / pandas / joblib / flask | 1.9.1 / 2.5.3 / 3.0.5 / 1.6.0 / 3.1.3 (iguais ao `requirements.txt`) |
| `.venv/` no Git | ignorado (`.gitignore:4: .venv/`) |
| `.python-version` | ainda diz `3.13` |

⚠️ O README do kit pede Python 3.12/3.13, e `.python-version` diz 3.13, mas o venv usa 3.14.10 e
todas as dependências instalaram. **Pendência:** na hospedagem, `PYTHON_VERSION` deve ser a mesma
do venv onde o modelo for treinado (etapa 03); verificar se Render/Streamlit oferecem 3.14.10, ou
retreinar em 3.13.

## 3. Resultados dos testes originais (linha de base)

`.venv/bin/python testar_sem_api.py` (executado nesta etapa, antes de qualquer mudança):
- Conferência da ponte: `{'ok': True, 'detalhes': []}`.
- Previsão do exemplo: 4199,5 (R$/mês), MAE 247,5, R² 0,944; casos semelhantes, alertas de faixa,
  erro de categoria, estatísticas por bairro, RAG e verificação (OK `[]`, FALHA `[9999.0]`) funcionando.

Hashes de referência (não podem mudar nesta etapa):

| Arquivo | SHA-256 |
|---|---|
| `dados/tech_layoffs.csv` | `a9ef120e81d7206de5be8a56337c0c008792363fec9ad65597610f9d32d81e6a` |
| `dados/dataset_legivel.csv` | `a2b88292f1c05ab9c9fae8ba84b170bd74a8b7e653485d531d4ab14a88c0f283` |
| `modelo/modelo.joblib` | `d1d40c71824e1e30ae8e5c2a29af6198f329de6f35edd7cd13026838a7d36113` |
| `modelo/metricas.json` | `a7723936fe8ad3ddd2299bf0fd865b65a25ad755a0fb431ec1140e63993bf163` |
| `modelo/amostra_conferencia.json` | `f7c82eebfc502e603bd7e0f5a862999b52d70652d243b29b570d786d82cda882` |

## 4. Objetivos da etapa

1. `preparar_dados.py`: funções importáveis (carregar, validar colunas, limpar, derivar, validar,
   separar X/y, exportar) + `main()`.
2. Gerar `dados/dataset_legivel_tech_layoffs.csv`.
3. Documentar o perfil real do dataset em `reports/PERFIL_DATASET.md`.
4. Testes `unittest` em `tests/test_preparar_dados.py`.
5. Confirmar que o kit continua funcionando e que o CSV original não mudou.

## 5. Análise da metodologia anterior (notebook)

| Passo | EDA (célula 5) | Treino (célula 7) |
|---|---|---|
| Colunas | `df.columns.str.strip()` + checagem de 17 colunas obrigatórias (célula 3) | idem (usa o mesmo `df`) |
| Duplicatas | — | `drop_duplicates()` (linhas idênticas) |
| Alvo | `to_numeric(coerce)` | `to_numeric(coerce)`; mantém `notna` e `>= 0` |
| Capital | `to_numeric(astype('string').str.replace(r'[^0-9.\-]', ''))` | `to_numeric(coerce)` direto |
| Porte anterior | `to_numeric(coerce)` | `to_numeric(coerce)` |
| Data | — | `to_datetime(coerce)` |
| `month` | — | `Date_layoffs.dt.month` |
| `year` | (usa `Year`) | `to_numeric(Year)` — **coluna `Year`, não o ano da data** |
| Entradas | — | cats `Country, Continent, Industry, Stage`; nums `Company_Size_before_Layoffs, Money_Raised_clean, year, month` |
| Ausentes | — | categóricas → `np.nan` (object); numéricas → `float64`; **imputação só no Pipeline** |

**Divergência EDA × treino no capital captado:** verificada no CSV real — os 1.635 valores não
nulos são idênticos nas duas formas e os 110 ausentes também; muda só o dtype (`Float64` nullable na
EDA, `float64` no treino). **Decisão:** usar a forma do **treino** (`pd.to_numeric` direto), pois é a
que gerou as métricas do modelo. O regex da EDA também quebraria notação científica (`1e5` → `15`).

## 6. Decisões técnicas

1. **Reaproveitar exatamente as regras do treino** (célula 7) na limpeza; nada de imputação, escala,
   encoding ou log nesta etapa.
2. **Não remover linhas por campos opcionais ausentes** (`Stage`, `Money_Raised_in__mil`, `USState`).
3. **Preservar todas as 18 colunas originais** + 3 derivadas (`Money_Raised_clean`, `year`, `month`).
   `Money_Raised_clean`/`Money_Raised_in__mil` e `year`/`Year` ficam com valores iguais neste CSV,
   mas são mantidas para compatibilidade 1:1 com o notebook.
4. **Não normalizar textos** (ex.: `UK` × `United Kingdom`, `Transportion`, `FreshBooks` × `Freshbooks`):
   o modelo anterior foi treinado com essas grafias; normalizar mudaria a metodologia. Fica
   documentado como inconsistência e pendência.
5. **Divergência `Year` × ano da data (7 linhas):** mantida como no notebook (`year` vem de `Year`);
   documentada.
6. **Legibilidade na exportação:** colunas de contagem com valores inteiros (`Laid_Off`, portes,
   `Year`, `year`, `month`, `Nr`) exportadas como inteiro com ausente preservado (`Int64`), data em
   `AAAA-MM-DD`, UTF-8, `\n`, sem índice. O valor numérico é o mesmo; a etapa 03 converte para
   `float64` em `separar_atributos_modelo()`, como no notebook.
7. **`separar_atributos_modelo()`** devolve `X, y` exatamente como a célula 7 (sem split), para o
   treino da etapa 03 reutilizar. Constantes `CATEGORICAS`, `NUMERICAS`, `ALVO`, `EXCLUIDAS_VAZAMENTO`.
8. **Mínimo de 1.000 linhas** (regra acadêmica do notebook) checado em `validar_dataset()` com
   parâmetro, para permitir testes com dados controlados pequenos.
9. `main()` aceita origem/destino para testes em diretório temporário; por padrão usa `dados/`.

## 7. Arquivos envolvidos

| Arquivo | Ação |
|---|---|
| `preparar_dados.py` | criar |
| `dados/dataset_legivel_tech_layoffs.csv` | gerar |
| `reports/PERFIL_DATASET.md` | criar |
| `tests/__init__.py`, `tests/test_preparar_dados.py` | criar |
| `plan/02-preparacao-dataset.md` | criar / atualizar |
| `plan/01-organizacao-projeto.md` | atualizar pendência do ambiente (com histórico) |
| `README.md` | atualizar a seção Tech Layoffs |
| `dados/tech_layoffs.csv`, `dados/dataset_legivel.csv`, `modelo/*`, código do kit | **não alterar** |

## 8. Checklist

- [x] Confirmar ambiente (`.venv`, versões, `.venv/` ignorado)
- [x] Rodar `testar_sem_api.py` (linha de base) e registrar hashes
- [x] Analisar o CSV com pandas e o notebook
- [x] Criar este plano
- [x] Atualizar pendência do ambiente no plano 01 (histórico preservado)
- [x] Implementar `preparar_dados.py`
- [x] Gerar `dados/dataset_legivel_tech_layoffs.csv`
- [x] Criar `reports/PERFIL_DATASET.md` com números reais
- [x] Criar `tests/test_preparar_dados.py` (12 grupos de teste → 20 testes)
- [x] Executar testes unitários (20/20 OK)
- [x] Reexecutar `testar_sem_api.py` e `conferir.py` (saídas iguais à linha de base)
- [x] Conferir hashes dos arquivos protegidos (todos iguais)
- [x] Atualizar README
- [x] Registrar `git status`

## 9. Estratégia de testes

- **Dados controlados** (DataFrames pequenos montados no teste): coluna ausente, data inválida,
  capital ausente/texto, alvo negativo/ausente, duplicatas, categorias com ausentes.
- **Dados reais**: carregamento (1.745 × 18), contagens de ausentes preservadas, compatibilidade de
  `X, y` com uma reprodução literal da célula 7 do notebook, exportação idêntica em duas execuções,
  hash do CSV original igual antes/depois e igual ao hash de referência.
- **Regressão do kit**: `testar_sem_api.py`, `conferir.py` e hashes de `modelo/` e `dataset_legivel.csv`.

## 10. Critérios de conclusão

Os critérios do enunciado (seção 15): ambiente confirmado, script implementado, dataset gerado,
original preservado, compatibilidade com o notebook, perfil documentado, testes executados, kit
funcionando, README e plano atualizados.

## 11. Resultados da execução

### `preparar_dados.py`
```
linhas_originais: 1745 | colunas_originais: 18 | duplicatas_removidas: 0
removidas_por_alvo_invalido: 0 | linhas_finais: 1745 | colunas_finais: 21 | datas_invalidas: 0
```
`dados/dataset_legivel_tech_layoffs.csv` — SHA-256
`dbe54cfd786fb57f393563b2ad76943f821f882349acccfc270251c413295909`, idêntico em duas execuções e
quando executado de outro diretório (`/tmp`).

### Testes unitários — `.venv/bin/python -m unittest discover -s tests -v`
`Ran 20 tests ... OK`

| Requisito | Teste(s) |
|---|---|
| 1. carregamento do original | `test_01_carrega_dataset_original` |
| 2. colunas obrigatórias ausentes | `test_02_coluna_obrigatoria_ausente`, `test_02b_carregar_remove_espacos_dos_nomes` |
| 3. datas | `test_03_conversao_de_datas`, `test_03_datas_reais_validas` |
| 4. `year` e `month` | `test_04_year_e_month` (inclui a divergência real `Year` × data) |
| 5. capital captado | `test_05_capital_captado` |
| 6. categóricos preservados | `test_06_categorias_preservadas`, `test_06_categorias_reais_preservadas` |
| 7. duplicatas | `test_07_duplicatas` |
| 8. alvo | `test_08_variavel_alvo` |
| 9. ausentes | `test_09_ausentes_nao_removem_linhas_nem_viram_zero`, `test_09_ausentes_reais_preservados`, `test_09b_validacao_minimo_de_linhas` |
| 10. compatibilidade com o notebook | `test_10_compatibilidade_com_notebook` (X/y iguais a uma cópia literal da célula 7), `test_10b_csv_legivel_reproduz_entradas_do_notebook`, `test_10c_exportacao_nao_altera_valores_originais` |
| 11. reprodutibilidade | `test_11_exportacao_reprodutivel` |
| 12. original preservado | `test_12_csv_original_preservado` |
| extra | `test_importar_nao_executa_processamento` |

### Regressão do kit
- `testar_sem_api.py`: exit 0, ponte `{'ok': True, 'detalhes': []}`, previsão 4199,5, verificação OK `[]` / FALHA `[9999.0]` — igual à linha de base.
- `conferir.py`: exit 0, ponte ok, 1500 linhas × 5 colunas (aluguel) — igual.
- Hashes de `dados/tech_layoffs.csv`, `dados/dataset_legivel.csv` e `modelo/*`: **SUCESSO** (inalterados); `tech_layoffs.csv` também idêntico à cópia em `dist/`.
- `git diff` vazio para `agente.py`, `app.py`, `ferramentas.py`, `recuperacao.py`, `prompts/`, `static/`, `conhecimento/`, `modelo/`, `notebooks/`, `reports/resultados/`.

### Divergências em relação ao notebook
1. Capital captado: EDA usa regex, treino usa `to_numeric` → adotado o do treino; valores idênticos neste CSV (ver §5).
2. Tipos na exportação: contagens gravadas como inteiro (`20` em vez de `20.0`) e `month`/`year` inteiros. Ao reler e passar por `separar_atributos_modelo()`, `X`/`y` são **idênticos** aos do notebook (teste 10b).
3. O notebook não preserva as colunas não usadas no `work`; o dataset legível mantém todas as 18 para consultas históricas.
Nenhuma divergência de valores.

## 12. Pendências para a etapa 03 e posteriores

1. Criar `treinar.py` usando `preparar_dados.preparar()` + `separar_atributos_modelo()` e o mesmo
   Pipeline/grade/seed do notebook; confirmar se as métricas reproduzem `metricas_teste.csv`
   (Colab × sklearn 1.9.1 podem diferir).
2. Escolher e justificar o modelo final (MLP pelo MAE ou KNN) e exportar `modelo/modelo.joblib`,
   `metricas.json` e `amostra_conferencia.json` reais.
3. Decidir sobre a **normalização de grafias** (`UK`/`United Kingdom`, `Transportion`…) e sobre as
   7 linhas com `Year` ≠ ano da data — qualquer mudança altera a metodologia e deve ir ao relatório.
4. Substituir `dados/dataset_legivel.csv` (ou apontar `ferramentas.py` para o novo arquivo) **junto**
   com modelo, `ferramentas.py` e `para_o_modelo()`.
5. Muitas categorias raras (30 setores e 25 países com < 5 eventos): avaliar efeito em
   `casos_semelhantes()` e nas mensagens de categoria inválida.
6. Compatibilidade de Python 3.14.10 com a hospedagem; `.python-version` diz 3.13.
7. Confirmar a versão do dataset no Kaggle e a moeda/unidade de `Money_Raised_in__mil`.
8. Usar `reports/PERFIL_DATASET.md` como fonte para `conhecimento/descricao_dataset.md` e
   `conhecimento/limitacoes.md`.
9. O comentário alterado em `requirements.txt` ("notebooks/Colab", pendência da etapa 01) foi
   incluído no commit `e8a2beb`; é só texto, sem efeito nas versões.
