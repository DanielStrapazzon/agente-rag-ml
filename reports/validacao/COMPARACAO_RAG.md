# Comparação com e sem RAG

> **Status: NÃO EXECUTADA** (aguardando autorização para consumir a cota do Gemini).
> Os resultados serão gravados por `comparar_sem_rag.py --executar` em `comparacao_rag_resultados.json`.

## Metodologia

Mesmas 5 perguntas, mesmo modelo (`gemini-3.5-flash-lite`), conversa nova em cada pergunta, três modos:

| Modo | O que tem | O que mede |
|---|---|---|
| a) Gemini puro | só a persona (`prompts/responder.md`) | linha de base sem dados |
| b) Agente sem RAG | pipeline completo, mas `RECUPERADOR.buscar` devolve `[]` | efeito das consultas determinísticas, do modelo e da verificação sem texto recuperado |
| c) Agente completo | RAG + consultas + modelo + verificação | sistema entregue |

- **(a) × (c)** mostra o efeito **conjunto** de RAG, ferramentas e verificação, não só do RAG.
- **(b) × (c)** isola a recuperação textual. Espera-se diferença maior nas perguntas 4 e 5 (metodologia e
  limitações), que dependem da base de conhecimento, e menor nas perguntas 1–3, que as consultas
  determinísticas respondem. **Isso é hipótese, não resultado:** registrar o que for observado, inclusive se o
  agente completo for pior em alguma pergunta.

## Gabaritos (pandas no CSV original / `metricas.json`)

| # | Pergunta | Gabarito |
|---|---|---|
| 1 | Quantas demissões estão registradas no dataset? | 581.705 funcionários demitidos (1.745 eventos) |
| 2 | Qual país registrou mais demissões? | USA (Estados Unidos): 425.063 |
| 3 | Quantas demissões ocorreram no Brasil em 2023? | 1.643 (9 eventos) |
| 4 | Qual algoritmo foi selecionado e por quê? | MLP; menor MAE na validação cruzada (201,19 × 210,19 do KNN) |
| 5 | Quais são as principais limitações do modelo? | não estima probabilidade de demitir; só eventos registrados; divisão aleatória; empresas repetidas; categorias raras; eventos extremos; casos semelhantes não são validação independente |

## Registro (preencher após a execução real)

Para cada pergunta e modo: resposta, etapas, fontes, tempo, número do gabarito presente (automático),
correção factual, completude e limitações citadas (revisão humana).

| # | Modo | Número correto? | Correção factual | Completude | Limitações citadas | Tempo (s) |
|---|---|---|---|---|---|---|
| 1–5 | a, b, c | não executado | — | — | — | — |

## Conclusão

Pendente da execução real. Não presumir que o agente completo é melhor em todas as perguntas.
