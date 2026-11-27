# Roteiro de apresentação (≈ 3 minutos, sem slides) e perguntas de estudo

> Antes: abrir `…/saude` (acordar a hospedagem), ter a chave reserva à mão, rodar `python conferir.py` para
> ter os números na tela. Usar **Nova conversa** entre os blocos.

## Demonstração

| Tempo | O que mostrar | O que dizer |
|---|---|---|
| 0:00–0:25 | Tela inicial (indicadores) | Problema: analisar demissões no setor de tecnologia. Dataset Tech Layoffs: 1.745 eventos, 1.327 empresas, 581.705 demitidos, 2020–2025. Os indicadores são calculados do CSV pelo servidor. |
| 0:25–0:50 | Painel "Como o agente responde" | 5 etapas: interpretar (Gemini → JSON), recuperar (TF-IDF + pandas), prever (MLP), gerar (Gemini), verificar números (Python). O Gemini nunca vê o CSV nem o modelo. |
| 0:50–1:20 | Sugestão "Compare as demissões de 2022 e 2023." | Consulta estatística em pandas (127.577 × 176.946, +38,7%); gráfico desenhado do resultado estruturado; abrir "Etapas executadas" e mostrar `consulta: comparacao`. |
| 1:20–1:45 | "Quais são as limitações do modelo?" | RAG: trechos de `conhecimento/limitacoes.md` aparecem nas fontes; sem consulta e sem modelo. |
| 1:45–2:25 | Previsão do teste 04 (EUA, Security, Series C, 800 funcionários, 150 milhões, março/2024) | Etapa `modelo: previsão`; estimativa ≈ 47,57; MAE 236,03 (não é intervalo de confiança); casos semelhantes (média 220,8, mediana 40). O modelo estima o tamanho de um evento, não a chance de demitir. |
| 2:25–2:45 | Resultado de `COMPARACAO_RAG.md` | Gemini puro × agente sem RAG × agente completo (citar só o que foi observado na execução real). |
| 2:45–3:00 | Encerramento | Limitações: dataset só com eventos registrados, divisão aleatória, empresas repetidas, eventos extremos, previsão pode superar o porte em empresas pequenas (o agente avisa). |

## Perguntas de estudo (respostas conforme o código do repositório)

**O que é RAG?** Recuperar informação relevante antes de gerar a resposta. Aqui: `recuperacao.py` busca trechos em
`conhecimento/*.md` e em fatos calculados do CSV (`ferramentas.fatos_do_dataset()`), e o agente junta o resultado
das consultas pandas; o Gemini redige usando só esse CONTEXTO.

**Como funciona o TF-IDF?** Cada trecho vira um vetor de pesos de palavras e bigramas (`TfidfVectorizer`,
`ngram_range=(1, 2)`, `sublinear_tf=True`, sem acentos, stopwords em português): palavras frequentes no trecho e
raras na base pesam mais. A pergunta é expandida com sinônimos e aliases (Brasil → Brazil, varejo → Retail).

**Similaridade de cosseno?** Ângulo entre os vetores da pergunta e de cada trecho (1 = mesma direção). Retornamos
até 4 trechos com nota ≥ 0,08, e nada se o melhor trecho ficar abaixo de 0,15 (pergunta fora da base).

**Por que pandas nas consultas?** Números exatos e reprodutíveis; o Gemini só escolhe a operação e os filtros num
JSON validado (`ferramentas.consultar_dados`, listas fechadas de operações/colunas/operadores, sem `eval`/SQL).

**Por que o Gemini não recebe o CSV inteiro?** Custo e limite de tokens, e principalmente confiabilidade: modelos
de linguagem erram contas; o código calcula e o Gemini só redige. Também evita expor dados sem necessidade.

**Como funciona a interpretação JSON?** A etapa 1 usa `response_json_schema` gerado a partir das listas de
`ferramentas.py`; campos opcionais vêm como `""`/lista vazia; `montar_consulta()` converte para o contrato de
`consultar_dados()`; tipo inválido vira erro controlado, nunca é descartado em silêncio.

**RAG × Machine Learning?** RAG recupera texto/fatos para fundamentar respostas; o ML (MLP) **estima** um número
(demitidos num evento) a partir de 8 atributos. São etapas diferentes do pipeline.

**Por que o MLP?** Menor MAE na validação cruzada de 5 dobras (201,19 contra 210,19 do KNN e 101.025,85 da
Regressão Linear); também menor MAE no teste (236,03). A vantagem sobre o KNN é pequena.

**Como funciona o Pipeline?** `TransformedTargetRegressor(log1p/expm1)` → `Pipeline(ColumnTransformer, MLP)`:
numéricas com mediana + `StandardScaler`; categóricas com mais frequente + `OneHotEncoder(handle_unknown="ignore")`;
ajustado só no treino de cada dobra. Por fora, `EstimadorNaoNegativo` troca previsões negativas por zero.

**Por que log1p no alvo?** `Laid_Off` é muito assimétrico (assimetria 9,55; mediana 90, máximo 22.000); o log reduz
o peso dos eventos gigantes no treino; `expm1` volta à escala original.

**Ponte `para_o_modelo()`?** Monta um DataFrame de 1 linha na ordem e nos tipos do treino (categóricas como texto,
numéricas float, ausente = NaN). Não escala nem codifica: isso está dentro do `.joblib`. A conferência refaz as
6 previsões de `amostra_conferencia.json` ao iniciar (`ok: True`).

**MAE, RMSE, R²?** MAE: erro absoluto médio no teste (236,03 funcionários) — não é intervalo de confiança.
RMSE (851,53): pesa mais os erros grandes. R² (0,5537): fração da variação explicada — não é % de acertos.

**Distância de Gower?** Para dados mistos: categóricas valem 0 (igual) ou 1 (diferente); numéricas |a−b|/amplitude
(porte e capital em log1p); atributos ausentes são ignorados e a média é renormalizada. "Próximo" = ≤ 0,20; se
faltar, completa e avisa `fallback`.

**Verificação numérica?** Extrai números e datas da resposta e confere contra o CONTEXTO gerado pelo sistema
(não contra a mensagem do usuário nem o histórico): 1% para inteiros, 0,2% para decimais ou meia unidade da
última casa. Falhou → pede nova resposta (1 vez) → se persistir, resposta montada pelo código ou ressalva.
Limitação: confere presença, não se o número foi atribuído ao grupo certo.

**Limitações do modelo?** Só eventos registrados (sem empresas que não demitiram); não estima probabilidade;
divisão aleatória (não temporal); 121 empresas do teste também no treino; categorias raras; erro grande em eventos
extremos (MAE 2.171,4 nos 18 eventos > 1.000); 14 de 349 previsões do teste acima do porte; moeda do capital não
confirmada; 7 eventos com Year ≠ data.

**O que fizemos além do kit?** Ver "O que mudamos em relação ao kit" no README: consultas estruturadas
(11 operações, filtros, ranking, comparação, correlação), aliases PT→EN, validação de domínio, Gower, RAG com
fontes/sinônimos/limiar, verificação revista e resposta determinística, histórico como conteúdo não confiável,
interface nova com etapas/fontes/gráficos, proteção dos artefatos, 169 testes, validador acadêmico.
