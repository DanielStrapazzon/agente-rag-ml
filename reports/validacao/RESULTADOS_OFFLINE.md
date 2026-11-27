# Resultados offline (etapa 07)

> Tudo abaixo foi **executado de fato**, mas **sem o Gemini real**. Nada aqui comprova a integração com a API.
> Data: 2026-11-26. Ambiente: `.venv`, Python 3.14.10, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5.

## 1. Suíte automatizada — `.venv/bin/python -m unittest discover -s tests -v`

| Arquivo | Testes | Resultado |
|---|---:|---|
| `test_preparar_dados.py` | 20 | OK |
| `test_machine_learning.py` | 22 | OK |
| `test_ferramentas.py` | 45 | OK |
| `test_recuperacao.py` | 8 | OK |
| `test_agente.py` (Gemini simulado) | 36 | OK |
| `test_app.py` (agente simulado) | 14 | OK |
| `test_frontend.py` | 17 | OK |
| `test_validacao.py` (novo, etapa 07) | 7 | OK |
| **Total** | **169** | **169 aprovados, 0 reprovados** |

Os 162 testes anteriores continuam passando, sem nenhuma alteração. Tempo: 1,7 s de testes (3,2 s com a
inicialização). Nenhum aviso (`Warning`) do Python; as linhas `INFO`/`WARNING` na saída são logs esperados do
agente e do servidor (ex.: "Erro do Gemini: código 429" num teste que simula esse erro).

## 2. Scripts do kit

- `testar_sem_api.py`: exit 0; ponte legível → modelo `{'ok': True, 'detalhes': []}`; erros para mês 13,
  porte −5, "12 abc", ano 1800, capital −1, Wakanda, USA + Europe, Reino Unido (ambíguo); previsão acima do
  porte sinalizada; RAG sem trechos para "receita de bolo".
- `conferir.py`: exit 0; métricas MLP (R² 0,5537; MAE 236,03; RMSE 851,53; MAE CV 201,19); 1.745 eventos,
  1.327 empresas, 581.705 demitidos; ponte `ok: True`.

## 3. Artefatos (SHA-256, iguais às etapas 03–06)

| Arquivo | SHA-256 |
|---|---|
| `dados/tech_layoffs.csv` | `a9ef120e81d7206de5be8a56337c0c008792363fec9ad65597610f9d32d81e6a` |
| `dados/dataset_legivel.csv` | `dbe54cfd786fb57f393563b2ad76943f821f882349acccfc270251c413295909` |
| `modelo/modelo.joblib` | `903684f62204ff59a50652859e6330cff472cec7d97ecbd31e6911c8dd549caf` |
| `modelo/metricas.json` | `048129701c4881f5126e33dc52ae49f262d839ac998d5a0e4af3f134d88afe71` |
| `modelo/amostra_conferencia.json` | `52b2b182aed3518a45a86c6407b13f939a86fefd25653462de73d8f2ec089fa4` |

## 4. Bateria SIMULADA — `.venv/bin/python validar_agente.py`

O pipeline real (ferramentas, modelo, RAG, verificação) roda com o Gemini substituído: a etapa 1 devolve a
**interpretação esperada** (`interpretacoes_simuladas` em `bateria_testes.json`) e a etapa 4 devolve um texto
montado do contexto. Isso confere ferramentas, etapas, números e regras — **não** a qualidade da interpretação
nem da redação do Gemini. Resultado: **28/28 com nota sugerida 1,0** (12 acadêmicos + 06b + 09b + 14 de robustez),
detalhes em `resultados_simulados.json`.

O avaliador automático também foi testado contra respostas **erradas** (`tests/test_validacao.py`): número
diferente do gabarito → 0,5; etapa de modelo indevida → 0,0; "999.999" no teste de injeção → 0,0; chave exposta
→ 0,0; probabilidade inventada ("73%") → 0,0; previsão sem os dados → 0,0; erro 429 → 0,0.

## 5. Validação visual (etapa 06)

Chromium headless em 1440×900, 820×1100, 390×844 com o agente simulado (ver `plan/06-frontend.md`).
