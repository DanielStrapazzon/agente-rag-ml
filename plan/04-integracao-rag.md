# Etapa 04 — Integração RAG e ferramentas do agente Tech Layoffs

Data: 2026-11-26. Base: commit `98a8383` (etapa 03). Implementação com apoio do **Claude Code**
(Anthropic); revisão e execução pelo grupo.

## 1. Contextualização

O kit ainda usa os artefatos de aluguel em `dados/dataset_legivel.csv` e `modelo/`. As etapas 02 e 03
produziram o dataset legível e o modelo MLP do Tech Layoffs em caminhos temporários. Esta etapa troca
os artefatos de forma coordenada com `ferramentas.py`, cria consultas estatísticas estruturadas,
monta a base de conhecimento RAG e adapta `testar_sem_api.py`/`conferir.py`. **Sem Gemini, sem
mudar prompts, frontend ou o fluxo do `agente.py`.**

## 2. Objetivos

1. Migrar os artefatos para os caminhos canônicos do kit e remover as cópias temporárias.
2. Adaptar `ferramentas.py` ao domínio, preservando os contratos usados por `agente.py`.
3. `consultar_dados(consulta)`: consultas estatísticas seguras (pandas, listas permitidas).
4. Validação de entradas do domínio, aliases PT→EN e casos semelhantes para dados mistos.
5. Base de conhecimento em 4 arquivos + fatos dinâmicos em trechos curtos; `recuperacao.py` com
   fonte/seção e sinônimos.
6. Testes (`test_ferramentas.py`, `test_recuperacao.py`) contra cálculos independentes no CSV.

## 3. Arquitetura atual — contratos usados por `agente.py`

| Símbolo | Uso no agente | Contrato a preservar |
|---|---|---|
| `F.BASE` | caminho dos prompts | `Path` da raiz |
| `F.ATRIBUTOS` | esquema JSON da etapa 1 e pedidos de dados | `{nome: {"tipo", "descricao", "valores"?}}`; `valores` obrigatório nas categóricas |
| `F.COLUNAS_CONSULTA`, `F.AGRUPAMENTOS` | `enum` do esquema JSON | listas de `str` |
| `F.METRICAS` | contexto (menos `sklearn_version`) | dict JSON |
| `F.faltando(atributos)` | pede dados faltantes | `list[str]` |
| `F.preparar(atributos)` | valida | `(entrada, erros, alertas)` |
| `F.FAIXAS[c]` para `c in NUMERICAS` | contexto de erro | `(min, max)` |
| `F.prever(entrada)` | previsão | dict com chave `"previsao"` |
| `F.casos_semelhantes(entrada, previsao)` | validação | dict JSON |
| `F.estatisticas(coluna, agrupar_por)` | intenção "dados" | dict JSON |
| `RECUPERADOR.buscar(texto)` | RAG | `list[dict]` com chave `"trecho"` |
| `fatos_do_dataset()` | usado por `recuperacao.py` | `list[str]` |

## 4. Arquitetura proposta

```text
preparar_dados.py   → constantes (ALVO, CATEGORICAS, NUMERICAS) + DESTINO = dados/dataset_legivel.csv
previsao_layoffs.py → PASTA_MODELO = modelo/; para_dataframe_modelo(); prever_demissoes()
ferramentas.py      → importa constantes; aliases; validação; prever() → prever_demissoes(MODELO);
                      casos_semelhantes() (Gower); consultar_dados(); estatisticas() → consultar_dados;
                      fatos_do_dataset() (trechos curtos)
recuperacao.py      → TF-IDF + cosseno (mantido); trechos com fonte/seção; expansão de sinônimos PT↔EN
conhecimento/       → descricao_dataset.md, analise_exploratoria.md, modelo_metricas.md, limitacoes.md
agente.py, app.py, prompts/, static/ → sem alteração
```

## 5. Migração dos artefatos

Antes (todos versionados; recuperáveis com `git show 98a8383:<arquivo>`):

| Arquivo | SHA-256 antes (aluguel) | Origem Tech Layoffs | SHA-256 Tech Layoffs |
|---|---|---|---|
| `dados/dataset_legivel.csv` | `a2b88292…0f283` | `dados/dataset_legivel_tech_layoffs.csv` | `dbe54cfd…295909` |
| `modelo/modelo.joblib` | `d1d40c71…36113` | `modelo_tech_layoffs/modelo.joblib` | `903684f6…49caf` |
| `modelo/metricas.json` | `a7723936…bf163` | `modelo_tech_layoffs/metricas.json` | `04812970…e71` |
| `modelo/amostra_conferencia.json` | `f7c82eeb…2cda882` | `modelo_tech_layoffs/amostra_conferencia.json` | `52b2b182…fa4` |

Passos: copiar → conferir hash → atualizar caminhos (`preparar_dados.DESTINO`,
`previsao_layoffs.PASTA_MODELO`, testes) → remover `dados/dataset_legivel_tech_layoffs.csv` e
`modelo_tech_layoffs/`. O modelo **não** é retreinado.

Risco registrado: `treinar_exemplo.py`, `exemplo_classificacao/treinar_classificacao.py` e
`exemplo_transformacao/treinar_fora_do_pipeline.py` gravam em `dados/` e `modelo/`; rodá-los
sobrescreve os artefatos definitivos (recuperáveis pelo Git ou com `preparar_dados.py` + `treinar.py`).

## 6. Validação das entradas (`preparar`)

| Situação | Exemplo | Resultado |
|---|---|---|
| Não informado | `None`, `""` | `faltando()` → o agente pede |
| Desconhecido declarado (só `Stage`, `Money_Raised_clean`) | "não sei", "desconhecido" | aceito → `None` → imputação do Pipeline + **alerta** |
| Desconhecido em atributo obrigatório | `Country` = "não sei" | **erro** (atributo obrigatório) |
| Inválido | "abc", "12 abc", NaN/inf, porte 10,5, mês 13, ano 1800, porte ≤ 0, capital < 0 | **erro** |
| País × continente incompatíveis | Brazil + Europe | **erro** (cada país tem 1 continente no dataset) |
| Continente omitido com país conhecido | Brazil | `Continent` derivado (South America) + alerta; não é pedido |
| Fora da faixa observada | porte 900.000; ano 2027 | **alerta** de extrapolação |
| Categoria fora do treino | `South Korea` | alerta (OneHotEncoder ignora) |

Conversão numérica estrita (casamento completo): `5000`, `5.000`, `5,5 mil` (porte); `300`,
`1,2 bilhão` → 1200 (capital, em milhões; sem assumir moeda); mês por número ou nome (`abril`, `apr`).

Decisão `Stage` desconhecido: o Pipeline imputa a categoria mais frequente do treino (**Post-IPO**),
diferente da categoria real `Unknown` do dataset (228 eventos). Exemplo (amostra 2): `Unknown` → 590,3;
desconhecido → 707,7. Segue-se a instrução (imputação + alerta que diz explicitamente "Post-IPO" e
sugere `Unknown`). Pendência para decisão do grupo.

## 7. Aliases

- Normalização: sem acento, minúsculas, pontuação → espaço.
- Dicionário explícito PT→EN para países, continentes, setores e estágios (`ALIASES` em `ferramentas.py`).
- **Consultas:** um alias pode agrupar grafias equivalentes do dataset (`Reino Unido` → `UK` +
  `United Kingdom`; `Transportion` + `Transportation`; `Logistic` + `Logistics`; empresas com caixa
  diferente, como `Clearco`/`ClearCo`). O resultado informa as grafias agrupadas.
- **Previsão:** só aceita um valor conhecido; alias ambíguo (`Reino Unido`) → erro com as opções.

## 8. Consultas estatísticas — `consultar_dados(consulta) -> dict`

Operações: `soma`, `contagem` (eventos), `contagem_distinta`, `media`, `mediana`, `minimo`,
`maximo`, `desvio_padrao`, `ranking`, `comparacao`, `correlacao`. Filtros: `eq`, `ne`, `in`,
`not_in`, `gt`, `gte`, `lt`, `lte`, `between`. Colunas e operadores em listas permitidas; sem
`eval`/`query`/SQL. Retorno sempre JSON estrito, com unidade, filtros aplicados (já resolvidos),
registros encontrados e observações (grafias agrupadas, divergência `Year` × data, ausência de
registros ≠ ausência de demissões). Detalhes e exemplos no README.

## 9. Casos semelhantes

Distância de **Gower** (dados mistos) sobre as 8 entradas: categóricas 0/1; numéricas
`|a−b|/amplitude`, com `log1p` em porte e capital (muito assimétricos); atributos ausentes em qualquer
lado são ignorados e os pesos renormalizados. Distância ∈ [0, 1]. "Próximo" = distância ≤ **0,20**
(calibrado nos 349 cenários do teste: o 5º vizinho tem distância mediana 0,145; com 0,15 só 54% dos
cenários têm 5 casos próximos, com 0,20 são 78%). Se houver menos que `quantidade` casos próximos,
completa com os mais próximos e marca `fallback` + `aviso_fallback`.

## 10. Base de conhecimento e RAG

4 arquivos em `conhecimento/`, parágrafos curtos, termos PT+EN, só fatos de `PERFIL_DATASET.md`,
`COMPARACAO_TREINAMENTO.md`, notebook, `metricas.json` e dataset. Fatos dinâmicos: um trecho por ano,
continente, país (top 10), setor (top 10), maiores eventos etc. Recuperação: TF-IDF 1–2 gramas,
cosseno, expansão de sinônimos na consulta, resultado com `trecho`, `similaridade`, `fonte`, `secao`.

## 11. Checklist

- [x] Ler planos, código e contratos; confirmar versionamento e hashes dos artefatos antigos
- [x] Criar este plano
- [x] Migrar artefatos e atualizar caminhos
- [x] Adaptar `ferramentas.py`
- [x] Escrever `conhecimento/` (4 arquivos) e remover o conteúdo de aluguel
- [x] Adaptar `recuperacao.py`
- [x] Adaptar `testar_sem_api.py` e `conferir.py`
- [x] Ajustar testes existentes aos novos caminhos
- [x] Criar `tests/test_ferramentas.py` e `tests/test_recuperacao.py`
- [x] Executar testes, `testar_sem_api.py`, `conferir.py`
- [x] Remover cópias temporárias após conferência por hash
- [x] Atualizar README e este plano

## 12. Testes

Comparação com cálculos independentes (pandas direto no CSV original, sem chamar a função testada):
soma, contagem, contagem distinta, filtros, agrupamentos, rankings, comparações, correlação. Mais
validação, aliases, previsões, casos semelhantes, JSON estrito e recuperação (perguntas reais e fora
de escopo).

## 13. Critérios de conclusão

Os do enunciado (§24).

## 14. Resultados

### Migração
Cópias conferidas com `cmp` (idênticas) antes e depois; `dados/dataset_legivel_tech_layoffs.csv` e
`modelo_tech_layoffs/` removidos. Hashes finais: `dados/dataset_legivel.csv` `dbe54cfd…`,
`modelo/modelo.joblib` `903684f6…`, `metricas.json` `04812970…`, `amostra_conferencia.json` `52b2b182…`
(iguais à etapa 03 → modelo não retreinado); `dados/tech_layoffs.csv` `a9ef120e…` (intacto).
Caminhos atualizados: `preparar_dados.DESTINO`, `previsao_layoffs.PASTA_MODELO`, docstring de
`treinar.py`, `tests/test_machine_learning.py` (verificação dos artefatos de aluguel substituída por
"artefatos definitivos = hashes da etapa 03" e "nenhum resquício de aluguel"); notas nos relatórios.

### Testes — `.venv/bin/python -m unittest discover -s tests -v`
**95 testes OK**: `test_preparar_dados` 20, `test_machine_learning` 22, `test_ferramentas` 45,
`test_recuperacao` 8. Duas falhas de recuperação apareceram na primeira execução ("qual país teve mais
demissões?" e "quais as métricas do modelo?") e foram corrigidas na origem (redação dos fatos por país;
sinônimo "métricas"), não no teste.

`testar_sem_api.py`: ponte `{'ok': True, 'detalhes': []}`; previsão da 1ª linha (USA, Retail, Series C,
porte 50, 2020-03) = 46,24 (real 20); 5 casos próximos, média 217,4; erros para mês 13, porte −5,
"12 abc", ano 1800, capital −1, Wakanda, USA+Europe, Reino Unido (ambíguo); porte 6 → 45,63
`previsao_inconsistente`. `conferir.py`: métricas MLP (0,5537 / 236,03 / 851,53 / 201,19), 1.745 eventos,
1.327 empresas, 581.705 demitidos.

Simulação offline do `agente.responder()` (Gemini substituído por respostas fixas, só num script
temporário): intenções dados, previsão, dados faltantes (Continent não é pedido) e erro de validação
percorrem o pipeline sem exceção.

### Consultas reais (conferidas contra pandas no CSV original)
Brasil: 7.029 demitidos em 48 eventos; Brasil 2023: 1.643 em 9 eventos; Brasil 2021: 0 eventos;
Reino Unido (UK + United Kingdom): 13.849 em 58 eventos; 2022 × 2023: 127.577 × 176.946 (+49.369,
+38,7%); top setores: Consumer 61.155, Retail 59.228, Hardware 57.000; Intel: máximo 22.000;
janeiro/2023 (Date_layoffs): 108 eventos; Spearman Laid_Off × porte: 0,798.

## 15. Pendências para a etapa 05

1. **Esquema da etapa 1 (`agente.py`)**: acrescentar ao JSON de interpretação um objeto `consulta`
   no formato de `consultar_dados()` (operacao, coluna, filtros, agrupar_por, ordem, limite,
   comparar, colunas), com `enum` de `OPERACOES`, `TIPOS_FILTRO`, `OPERADORES`, `AGRUPAMENTOS`;
   na etapa 2 chamar `F.consultar_dados()` (mantendo `estatisticas()` como fallback).
2. `agente.py` rotula "modelo: previsão (extrapolação)" sempre que há **qualquer** alerta (inclusive
   "Continent deduzido"); separar alertas de extrapolação dos demais.
3. Incluir `fonte`/`secao` dos trechos no CONTEXTO (hoje o agente usa só `trecho`).
4. `prompts/responder.md`: persona Tech Layoffs; regras: MAE ≠ intervalo de confiança, R² ≠ acurácia,
   não é probabilidade de demissão, avisar `previsao_inconsistente`, `fallback`, grafias agrupadas,
   ausência de registros ≠ ausência de demissões. `prompts/interpretar.md`: explicar "desconhecido"
   (Stage/capital) × não informado.
5. Decidir sobre **Stage desconhecido**: imputação (Post-IPO, implementado conforme o enunciado) ×
   mapear para a categoria real `Unknown` (ex.: amostra 2 → 707,7 × 590,3).
6. Frontend (`static/index.html`) e `versao_streamlit/streamlit_app.py`: textos de aluguel.
7. `comparar_sem_rag.py`: perguntas de aluguel (usa a API).
8. Limite do RAG: perguntas fora do escopo com palavras do domínio ("cotação do dólar", "melhor filme
   de 2023") ainda recuperam trechos; o filtro principal é a intenção `fora_escopo` da etapa 1.
9. Hospedagem: Python 3.14.10 × `.python-version` 3.13.
10. Risco: scripts de exemplo do kit (`treinar_exemplo.py` etc.) sobrescrevem `dados/` e `modelo/`.
