# Registro do uso de inteligência artificial (rascunho para o relatório)

> Rascunho preparado na etapa 07. As seções marcadas **[GRUPO PREENCHE]** só podem ser completadas pelos
> integrantes: este documento não atribui ao grupo atividades que não estão registradas no repositório.

## Ferramentas

| Ferramenta | Uso declarado | Evidência no repositório |
|---|---|---|
| **Claude Code** (Anthropic), modelo Claude | implementação assistida das etapas 01 a 07: código, testes, documentação, planos | cabeçalhos "com apoio do Claude Code" nos arquivos; `plan/01` a `plan/07`; README ("O que mudamos em relação ao kit") |
| **ChatGPT** (OpenAI) | planejamento e redação dos enunciados de cada etapa (prompts enviados ao Claude Code), revisão das entregas; no trabalho anterior, apoio na estruturação e revisão (ver `reports/ROTEIRO_RELATORIO.md`) | declarado pelo grupo; os enunciados das etapas não estão versionados **[GRUPO CONFIRMA]** |
| **Gemini** `gemini-3.5-flash-lite` | componente do próprio agente (interpretar e redigir), exigido pelo professor | `agente.py` — não foi usado para escrever código nem para gerar gabaritos |

## Arquivos produzidos ou alterados com apoio do Claude Code

| Etapa | Arquivos | Funcionalidade |
|---|---|---|
| 01 | `plan/01-organizacao-projeto.md`, `.gitignore`, README (seção do grupo); cópias de `dist/` para `dados/`, `notebooks/`, `reports/`, `referencias/` | organização e análise do trabalho anterior |
| 02 | `preparar_dados.py`, `tests/test_preparar_dados.py`, `reports/PERFIL_DATASET.md`, `plan/02-…` | preparação reproduzível do dataset |
| 03 | `treinar.py`, `previsao_layoffs.py`, `tests/test_machine_learning.py`, `reports/novo_treinamento/*`, `plan/03-…` | reprodução do treino, exportação do MLP |
| 04 | `ferramentas.py`, `recuperacao.py`, `conhecimento/*.md`, `testar_sem_api.py`, `conferir.py`, `tests/test_ferramentas.py`, `tests/test_recuperacao.py`, `plan/04-…`; migração de `dados/` e `modelo/` | consultas estruturadas, validação, aliases, Gower, base RAG |
| 05 | `agente.py`, `prompts/interpretar.md`, `prompts/responder.md`, `app.py`, `protecao_artefatos.py` (+ guarda nos 3 scripts de exemplo), `comparar_sem_rag.py`, `tests/test_agente.py`, `tests/test_app.py`, `plan/05-…` | agente conversacional, verificação, segurança |
| 06 | `static/index.html`, `static/css/app.css`, `static/js/app.js`, `static/assets/favicon.svg`, `app.py` (`/api/dataset/resumo`), `agente.py` (2 correções), `versao_streamlit/streamlit_app.py`, `tests/test_frontend.py`, `plan/06-…` | interface |
| 07 | `validar_agente.py`, `comparar_sem_rag.py` (gravação JSON), `tests/test_validacao.py`, `reports/validacao/*`, `plan/07-…` | validação acadêmica |

Não foram gerados com IA: o dataset (`dados/tech_layoffs.csv`), o notebook e os resultados do trabalho anterior
(`notebooks/`, `reports/resultados/` — ver a declaração daquele trabalho) e o kit do professor.

## Decisões e atividades do grupo

Registradas no repositório ou nas conversas de desenvolvimento:
- definição do escopo e dos requisitos de cada etapa (enunciados 01–07) e das restrições (não retreinar, não
  alterar o CSV, não consumir a API sem autorização, não fazer commits automáticos);
- criação do ambiente `.venv` (Python 3.14.10) e instalação das dependências; execução inicial de
  `testar_sem_api.py` e `conferir.py` (etapa 02);
- commits de cada etapa (`restruc:`, `feat: dataset`, `feat: machine learning`, `feat: integration rag`,
  `feat: integration agent`, `feat: frontend`).

Decisões técnicas tomadas pelo assistente e **aceitas** pelo grupo ao seguir para a etapa seguinte
**[GRUPO CONFIRMA cada uma]**: seleção do MLP pelo MAE de validação cruzada; manutenção da imputação do
Pipeline para Stage desconhecido (em vez da categoria `Unknown`); limiar de Gower 0,20; agrupamento de grafias
só nas consultas; troca do setor "Enterprise" por "Security" no teste 04.

## Revisões humanas efetivamente realizadas — [GRUPO PREENCHE]

- [ ] Quem leu e entendeu cada arquivo de código (apresentação/arguição exige explicar tudo).
- [ ] Testes executados pelos integrantes (comandos, datas, resultados).
- [ ] Execução real da bateria e da comparação com/sem RAG e revisão das notas.
- [ ] Ajustes feitos manualmente no código ou nos textos.
- [ ] Participação de cada integrante (o histórico do Git mostra um único autor nos commits do grupo).
