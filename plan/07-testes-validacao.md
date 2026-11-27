# Etapa 07 — Testes, validação acadêmica e avaliação do agente

Data: 2026-11-26. Base: commit `69af5c7` (etapa 06, 162 testes). Apoio do **Claude Code** (Anthropic).

## 1. Contextualização

O agente, as ferramentas, o RAG, o modelo e a interface estão implementados e testados **offline**
(Gemini simulado). Esta etapa organiza a validação acadêmica: gabaritos independentes, bateria de
12 testes, scripts para execução real controlada, comparação com/sem RAG, roteiro de apresentação,
checklist de entrega e registro do uso de IA. **Chamadas reais ao Gemini só com autorização explícita.**

## 2. Requisitos acadêmicos (fonte: README original do kit, seção "Testem como o professor vai testar")

- `python conferir.py` monta o gabarito do professor.
- Registrar no relatório **pelo menos 10 testes** (individual: 6); ideal cobrir as **12 categorias**:
  dados (3); previsão completa; previsão faltando 2 dados; valor absurdo; "essa previsão faz sentido?";
  fora do assunto; tentativa de revelar instruções/"esquecer as regras"; pergunta de acompanhamento;
  "explique para um leigo"; "esse modelo é confiável?".
- Previsão nova precisa passar pelo modelo (🔧 `modelo: previsão`); repetir número correto de mensagem
  anterior vale.
- Comparação com e sem RAG para o relatório (`comparar_sem_rag.py`).
- Commits "ao longo do trabalho, pelos dois integrantes"; `links.txt`; `PYTHON_VERSION` = versão do venv
  do treino; congelamento a partir de 02/12, 23h59.
- **Não encontrados na documentação local** (vêm do briefing da etapa 07, a confirmar com o professor):
  escala 1,0 / 0,5 / 0,0 por teste e limite de tempo por resposta. Tolerância numérica: o kit usa 1%
  para inteiros e 0,2% para decimais (docstring de `verificar()`); o briefing cita 1%.
- O enunciado original não existe como arquivo separado no repositório (o commit inicial "Kit inicial do
  agente + enunciado" contém o README do kit).

## 3. Estratégia de testes

1. Offline: suíte `unittest` (162), `testar_sem_api.py`, `conferir.py`, hashes dos artefatos.
2. Gabaritos calculados por `validar_agente.py` com pandas direto no CSV original e `modelo.predict()`
   com a ponte validada (independentes das funções avaliadas, exceto casos semelhantes — ver §4).
3. Bateria de 12 testes + robustez em `reports/validacao/bateria_testes.json`.
4. Modo offline do validador: pipeline real com o Gemini **simulado** devolvendo a interpretação esperada
   (verifica ferramentas, etapas e números; não verifica a interpretação do Gemini).
5. Modo real (`--executar`): smoke test primeiro; depois a bateria, com limite de chamadas e retomada.

## 4. Gabaritos

`reports/validacao/GABARITO.md` (gerado). Totais, por país/setor/ano, Brasil 2023/2024, 2022 × 2023,
correlações (Pearson e Spearman), métricas (lidas de `metricas.json` e MAE/R² recalculados no teste),
previsões de cenários controlados. Casos semelhantes usam `ferramentas.casos_semelhantes()` (não há
implementação independente de Gower) — registrado como limitação.

Ajuste do teste 04: o setor "Enterprise" do exemplo **não existe** no dataset (seria recusado); usado
**Security** (no treino; 49 eventos nos EUA, 5 em Series C); nenhuma linha do dataset é idêntica ao cenário.

## 5. Critérios de aprovação

Por teste: 1,0 correto / 0,5 parcial / 0,0 incorreto (escala do briefing). O validador sugere uma
nota automática (números do gabarito presentes com tolerância 1%, etapas esperadas presentes, etapas
proibidas ausentes, termos obrigatórios/proibidos); a nota final é **revisão humana**.

## 6. Controle de chamadas ao Gemini

Estimativa: 1 chamada (fora do escopo/dados faltando/esclarecimento), 2 (normal), 3 (verificação refeita).
Bateria completa ≈ 14 mensagens ≈ 30–40 chamadas; comparação com/sem RAG ≈ 5 × 5 = 25 chamadas.
`--max-chamadas` interrompe antes de exceder; resultados salvos a cada teste; `--continuar` pula os já
executados. Erros 403/429/503/timeout/interpretação/ferramenta/verificação registrados separadamente.

## 7. Comparação com e sem RAG

`comparar_sem_rag.py --executar` (3 modos, mesmas 5 perguntas, mesmo modelo); salva JSON com respostas,
etapas, fontes, tempos e conferência dos números do gabarito. Análise em `reports/validacao/COMPARACAO_RAG.md`.

## 8. Evidências

```text
reports/validacao/
├── ROTEIRO_TESTES.md          como executar (offline e real)
├── GABARITO.md                gerado pelo validador
├── bateria_testes.json        12 testes + robustez, com gabaritos e critérios
├── RESULTADOS_OFFLINE.md      suíte + validador offline (simulado)
├── RESULTADOS_VALIDACAO.md    resultados reais (pendente até haver execução autorizada)
├── COMPARACAO_RAG.md          metodologia + resultados (pendente)
├── ROTEIRO_APRESENTACAO.md    demonstração de 3 min + perguntas de estudo
├── CHECKLIST_ENTREGA.md
└── USO_DE_IA.md
```

## 9. Problemas encontrados

1. O exemplo do teste 04 usa o setor "Enterprise", que não existe no dataset; trocado por "Security" e coberto por
   `test_validacao.test_previsao_do_gabarito_usa_o_modelo`.
2. O avaliador automático daria 0,5 a uma probabilidade inventada sem etapa de modelo; criado o critério crítico
   "nenhum percentual" no teste 12 (`test_validacao.test_avaliador_reprova_respostas_erradas`).
3. Entrega: os 6 commits do grupo têm um único autor (o kit pede commits dos dois integrantes); `links.txt` ainda é
   o modelo; a versão do Python diverge (venv 3.14.10 × `.python-version` 3.13); a aplicação ainda não foi publicada.
Nenhuma falha nova do agente apareceu offline.

## 10. Checklist

- [x] Ler planos, código, prompts, relatórios, testes, `links.txt`; procurar o enunciado
- [x] Criar este plano
- [x] Validação offline inicial (suíte, scripts, hashes)
- [x] `validar_agente.py` (gabaritos, bateria, modo offline simulado, modo real controlado)
- [x] `comparar_sem_rag.py` grava resultados em JSON
- [x] Documentos em `reports/validacao/`
- [x] Revisão de hospedagem
- [x] README e este plano

## 11. Resultados

- Offline: **169/169** testes (162 anteriores sem alteração + 7 novos); `testar_sem_api.py` e `conferir.py` com
  exit 0; ponte `ok: True`; hashes de `dados/` e `modelo/` iguais às etapas anteriores.
- Validador offline (simulado): **28/28** com nota sugerida 1,0; o avaliador reprova as respostas erradas dos testes.
- Gabaritos: 581.705 demitidos; 1.745 eventos; 1.327 empresas (1.323 sem distinguir maiúsculas); Brasil 2023 =
  1.643 (9 eventos), 2024 = 75 (1 evento); 2022 × 2023 = +49.369 (+38,7%); Spearman do porte = 0,798 (Pearson 0,694);
  teste 04 = 47,57 com 5 casos próximos (média 220,8); MAE recalculado = 236,03.
- Real: **não executado** (sem autorização). A comparação com/sem RAG também não foi executada.
- Repositório público (API do GitHub responde 200 sem autenticação).

## 12. Pendências (etapa 08)

1. Autorizar e executar: `validar_agente.py --executar --smoke` → bateria (`--continuar`) → `comparar_sem_rag.py --executar`
   → revisão humana das notas em `RESULTADOS_VALIDACAO.md` e análise em `COMPARACAO_RAG.md`.
2. Decidir a versão do Python da hospedagem (3.14.10 ou retreino autorizado em 3.13) e publicar; preencher `links.txt`.
3. Commits do segundo integrante; completar `USO_DE_IA.md`; chave reserva; relatório final.

## 13. Critérios de conclusão

Preparação técnica: offline aprovado, bateria e gabaritos prontos, scripts prontos, evidências e
pendências documentadas. Validação acadêmica completa **só** após execução real autorizada e revisão humana.
