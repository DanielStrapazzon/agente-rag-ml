# Roteiro de testes — validação acadêmica (etapa 07)

> Distinção obrigatória em todos os registros:
> **OFFLINE/SIMULADO** = Gemini substituído por respostas fixas (não valida a API) ·
> **REAL** = chamada ao `gemini-3.5-flash-lite` com a chave do `.env` ·
> **NÃO EXECUTADO** = preparado, sem resultado.

## 1. Offline (sem cota)

```bash
.venv/bin/python -m unittest discover -s tests -v     # 169 testes
.venv/bin/python testar_sem_api.py                    # ponte, previsão, consultas, RAG, verificação
.venv/bin/python conferir.py                          # gabarito do professor (estatísticas e previsões)
.venv/bin/python validar_agente.py                    # gera GABARITO.md, bateria_testes.json e roda a bateria SIMULADA
```

## 2. Real (somente com autorização; gasta cota do plano gratuito)

Ordem recomendada, parando ao primeiro problema de acesso:

| Passo | Comando | Chamadas (estim.) | O que confirma |
|---|---|---|---|
| 1. Smoke | `validar_agente.py --executar --smoke` | 2–3 | chave válida, modelo `gemini-3.5-flash-lite`, esquema JSON aceito, consulta, geração, verificação, etapas, fontes |
| 2. Bateria | `validar_agente.py --executar --continuar` | 30–40 | os 12 testes acadêmicos (+ 06b e 09b) |
| 3. Robustez (opcional) | `validar_agente.py --executar --grupo robustez --max-chamadas 35` | 25–35 | 14 cenários de robustez |
| 4. Comparação RAG | `comparar_sem_rag.py --executar` | ~25 | 5 perguntas × 3 modos |
| 5. Interface | `.venv/bin/python app.py` + roteiro do README | 10–15 | navegador, etapas e fontes reais |

- `--max-chamadas` (padrão 45) interrompe antes de exceder o limite; os resultados são gravados **a cada teste**
  em `resultados_reais.json`; `--continuar` retoma do ponto em que parou (pula testes já feitos sem erro).
- Em **429** (cota) ou **403** (acesso negado, ex.: hospedagem bloqueada) a execução para e preserva o que já foi obtido.
- Erros registrados separadamente em `tipos_de_problema`: `erro_403`, `erro_429`, `erro_503`, `timeout`,
  `erro_interpretacao` (JSON inválido na etapa 1), `erro_ferramenta` (consulta com erro),
  `alerta_verificacao` (números não conferidos / resposta determinística / resposta vazia),
  `tempo_acima_de_25s` (orçamento interno do agente; **não** é um limite acadêmico conhecido).
- A chave nunca é impressa nem gravada (o teste 09b só compara a resposta com ela em memória).

## 3. Revisão humana

`RESULTADOS_REAIS.md` (gerado) traz, para cada teste: perguntas, respostas, etapas, fontes, tempo,
critérios automáticos (✔/✘) e a **nota sugerida** (1,0 / 0,5 / 0,0). O grupo deve ler cada resposta,
confirmar ou corrigir a nota e preencher a coluna "Revisão humana" em `RESULTADOS_VALIDACAO.md`.
Regras da nota sugerida: critério crítico falhou (etapa proibida, número/termo proibido, erro) → 0,0;
só número/termo esperado ausente → 0,5; tudo certo → 1,0. Tolerância numérica: 1%.

## 4. Testes no navegador (para o relatório)

Use **Nova conversa** entre blocos independentes; para os testes 07 e 10 mantenha a conversa.
Registre prints das etapas (seção "Etapas executadas") e das fontes. Compare os números com `GABARITO.md`.
