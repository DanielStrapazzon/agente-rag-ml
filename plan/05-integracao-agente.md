# Etapa 05 — Integração do agente conversacional com Gemini

Data: 2026-11-26. Base: commit `d911e4c` (etapa 04, 95 testes). Implementação com apoio do
**Claude Code** (Anthropic); revisão e execução pelo grupo.

## 1. Contextualização

As ferramentas determinísticas (`ferramentas.py`: `consultar_dados`, `preparar`, `prever`,
`casos_semelhantes`) e o RAG (`recuperacao.py`) já funcionam offline. O `agente.py` e os prompts
ainda são os do kit (aluguel): a etapa 1 só sabe pedir `estatisticas(coluna, agrupar_por)`.
Esta etapa liga o agente às ferramentas novas, reescreve os prompts e endurece a verificação e a
segurança. **Sem chamadas reais ao Gemini sem autorização do usuário**; testes com `unittest.mock`.

## 2. Arquitetura atual (kit)

`interpretar` (Gemini JSON: intencao, atributos, consulta{coluna, agrupar_por}, busca) →
`RECUPERADOR.buscar` (só `trecho` vai ao contexto) → `estatisticas` / `preparar`+`prever`+`casos` →
`_gemini` (gera) → `verificar` (números da resposta vs. contexto **+ mensagem + histórico "model"**;
ignora inteiros < 10; converte qualquer decimal ≤ 1 em %). Histórico do cliente vira turnos
`user`/`model` reais da API (o cliente pode forjar falas do "model").

## 3. Arquitetura proposta

```text
mensagem + histórico (sanitizado: papéis válidos, ≤ 20 itens, ≤ 2.000 caracteres cada)
 │
 ├─1─ INTERPRETAR  Gemini (JSON Schema montado de ferramentas.py) → intencao, atributos,
 │                  consulta{operacao, coluna, agregacao, filtros[], agrupar_por[], ordem, limite,
 │                  comparar_coluna, comparar_valores[], colunas[], metodo}, busca, esclarecimento,
 │                  sobre_previsao_anterior
 │                  → normalização em Python (montar_consulta): strings vazias = ausente
 ├─2─ RECUPERAR    RECUPERADOR.buscar(busca) → [{trecho, fonte, secao}]   (similaridade só no log)
 │                  + F.consultar_dados(consulta)  (intenção "dados")
 ├─3─ PREVER       F.faltando → F.preparar → F.prever → F.casos_semelhantes  (alertas classificados)
 ├─4─ GERAR        Gemini com prompts/responder.md + CONTEXTO mínimo por intenção (JSON estrito)
 └─5─ VERIFICAR    números/datas da resposta ⊂ CONTEXTO (só fontes confiáveis); 1 nova tentativa;
                   se persistir: resposta determinística segura (dados/previsão) ou ressalva
```

Histórico e mensagem vão ao Gemini **dentro de um único turno `user`**, rotulados como conteúdo não
confiável — o cliente não consegue forjar falas do assistente.

## 4. Fluxo de interpretação

- `fora_escopo` → resposta fixa (1 chamada).
- `esclarecimento` não vazio → devolve a pergunta de esclarecimento (1 chamada).
- `dados` → `montar_consulta()` → `F.consultar_dados()`; sem operação → só RAG.
- `previsao` → faltando? pergunta só o que falta (Continent é deduzido do país).
- `explicacao` → RAG + métricas resumidas; previsão **só** se `sobre_previsao_anterior` e atributos completos.
- Perguntas subsequentes: o Gemini recebe o histórico e reconstrói a consulta completa
  ("E em 2024?" → mesma operação e país, ano trocado). Números sempre recalculados pelo Python.

## 5. Contrato JSON das consultas

Esquema da etapa 1 gerado a partir de `F.OPERACOES`, `F.AGREGACOES`, `F.COLUNAS_CONSULTA`,
`F.AGRUPAMENTOS`, `F.TIPOS_FILTRO`, `F.OPERADORES` (fonte de verdade única). Decisão por limitação
do JSON Schema do Gemini (sem uniões de tipos/`null`): todo campo opcional é string com `""` = ausente,
listas vazias = ausente, `limite` 0 = padrão, filtro com `valor` (string) **e** `valores` (lista de
strings). `montar_consulta()` converte para o contrato real de `consultar_dados()`
(`comparar_coluna`+`comparar_valores` → `comparar{coluna, valores}`; `valores` → `valor` em
`in/not_in/between`). Números chegam como texto e são convertidos pela conversão estrita de `ferramentas`.

## 6. Gerenciamento de contexto

CONTEXTO por intenção (JSON estrito, sem NaN): `intencao`, `pergunta_interpretada`,
`base_de_conhecimento` [{trecho, fonte, secao}], e
- dados: `consulta_executada`, `resultado_estatistico`;
- previsão: `entrada_usada`, `previsao_do_modelo`, `casos_semelhantes`, `alertas` (por tipo),
  `metricas_do_modelo` (resumidas);
- explicação: `metricas_do_modelo` resumidas + comparação dos 3 algoritmos;
- erro de entrada: `erros`, faixas e opções só dos atributos com erro.
Sem versões de bibliotecas, hashes, seeds ou similaridade.

## 7. Verificação numérica (melhorias)

| Problema do kit | Solução |
|---|---|
| inteiros < 10 ignorados | todos verificados; ordinais (1º, 2ª) ignorados |
| mensagem e histórico do cliente aceitos como evidência | só o CONTEXTO gerado pelo sistema |
| decimal ≤ 1 vira % mesmo sem "%" | conversão só quando a resposta escreve "%" |
| datas viram números soltos (2023, 01, 20) | datas ISO/BR verificadas como datas |
| números dentro de nomes ("7shifts") | ignorados (dígitos colados em letras) |
| sinal "−" (U+2212) | normalizado |
| arredondamento | tolerância do kit (1% inteiros, 0,2% decimais) **ou** meia unidade da última casa escrita |
Limitação: a verificação confere a **presença** dos números no contexto, não se cada número foi
atribuído ao grupo certo.

## 8. Segurança

`app.py` revisado: limites de tamanho (64 KB / 1.000 caracteres), rate limiting, JSON inválido → 400,
erros do Gemini → 503, falha inesperada → 500 sem detalhes. Novo: histórico sanitizado no agente;
consultas malformadas → erro controlado (sem 500); sem `eval`/SQL; chave só em `GEMINI_API_KEY`;
log sem a chave e com a pergunta truncada. Proteção dos artefatos: `protecao_artefatos.py` impede os
scripts de exemplo de sobrescrever `dados/` e `modelo/` do Tech Layoffs.

## 9. Testes

`tests/test_agente.py` e `tests/test_app.py` com `unittest.mock` (Gemini simulado; ferramentas reais).
Os mocks **não** comprovam que a API real funciona.

## 10. Checklist

- [x] Ler planos, código e contratos
- [x] Criar este plano
- [x] `ferramentas.py`: classificar alertas; endurecer `consultar_dados` contra tipos inesperados
- [x] `agente.py`: esquema, `montar_consulta`, histórico, contexto, verificação, resposta segura
- [x] `prompts/interpretar.md` e `prompts/responder.md`
- [x] `app.py`: devolver `fontes`; revisão de segurança
- [x] `protecao_artefatos.py` + guarda nos 3 scripts de exemplo
- [x] `comparar_sem_rag.py` (Tech Layoffs, 3 modos, sem executar)
- [x] Testes novos + execução completa; `testar_sem_api.py`; `conferir.py`; hashes
- [x] README e este plano

## 11. Critérios de conclusão

Os do enunciado (§26). A validação real com o Gemini é registrada à parte.

## 12. Resultados

### Testes — `.venv/bin/python -m unittest discover -s tests -v`
**145 testes OK**: os 95 anteriores (sem alteração) + `test_agente.py` 36 + `test_app.py` 14.
Nenhum teste anterior foi removido ou adaptado.

Os testes novos encontraram dois problemas, corrigidos na origem:
1. **Bug real em `montar_consulta()`:** `filtros` com tipo errado (ex.: a string "Brasil") era
   descartado em silêncio e a soma era feita sobre o dataset inteiro (581.705 em vez de erro).
   Agora o campo malformado segue para `consultar_dados()`, que devolve erro controlado.
2. O teste da proteção dos exemplos achava `joblib.dump(` na docstring do script: corrigido o teste
   (busca só no código). A guarda já estava antes de qualquer gravação.

Também endurecido `consultar_dados()`: valores de filtro em lista aninhada (`[["Brazil"]]`) eram aceitos;
estruturas inesperadas (TypeError/AttributeError) agora viram `{"erro": ...}`.

`testar_sem_api.py` e `conferir.py`: exit 0, ponte `{'ok': True, 'detalhes': []}`, métricas do MLP
iguais (0,5537 / 236,03 / 851,53 / 201,19). Hashes de `dados/tech_layoffs.csv`, `dados/dataset_legivel.csv`
e `modelo/*` iguais aos da etapa 04 (também verificados em `tests/test_app.py`).

### Integração real com o Gemini
**Não executada.** O `.env` existe, mas o enunciado exige autorização para consumir a API; nenhum teste
chama o Gemini. Pendente: rodar o roteiro de smoke test (README, "Executar localmente") e
`comparar_sem_rag.py --executar`, registrando etapas (🔧) e números conferidos com `conferir.py`.
Risco específico a validar: se a API aceita o `response_json_schema` (objetos aninhados em arrays,
`required` nos itens de filtro) e se o `gemini-3.5-flash-lite` preenche corretamente a consulta.

## 13. Limitações

- Os testes com mock validam a orquestração, não a qualidade da interpretação do Gemini.
- A verificação numérica confere presença dos números no contexto, não a associação número ↔ grupo.
- Perguntas subsequentes dependem de o Gemini reconstruir a consulta a partir do histórico (o servidor
  não guarda estado; o navegador envia o histórico).
- `Stage` desconhecido → imputação Post-IPO (metodologia do Pipeline); a categoria real `Unknown`
  daria outra estimativa. Decisão mantida e documentada; o usuário pode informar "Unknown".
- O prompt de interpretação lista todas as opções de país (42) e setor (60): mais tokens por chamada.
- O RAG lexical ainda recupera trechos para perguntas fora do escopo com palavras do domínio; o
  filtro principal é a intenção `fora_escopo`.

## 14. Pendências para a etapa 06

1. Validação real com o Gemini (roteiro do README) e ajustes de prompt conforme o resultado.
2. Interface: `static/index.html` e `versao_streamlit/streamlit_app.py` (título, boas-vindas,
   exemplos de pergunta; exibir `fontes`, que a API já devolve).
3. Hospedagem: Python 3.14.10 × `.python-version` 3.13; confirmar no Render/Streamlit.
4. Registrar no relatório a comparação com/sem RAG (`comparar_sem_rag.py --executar`) e a bateria
   de testes do professor (12 categorias).
