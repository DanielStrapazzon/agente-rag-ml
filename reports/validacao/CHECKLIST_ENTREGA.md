# Checklist de entrega (etapa 07)

Legenda: ✅ verificado · ⚠️ parcial / com ressalva · ⏳ pendente · ❓ a confirmar com o grupo/professor.
Verificado em 2026-11-26 sobre o commit `69af5c7` + alterações não commitadas da etapa 07.

| Item | Situação | Evidência / observação |
|---|---|---|
| Aplicação web funcional | ⚠️ | Flask + interface testados offline (169 testes, Chromium headless com agente simulado); **não testada com o Gemini real** |
| Modelo Gemini exigido | ✅ | `agente.MODELO_GEMINI = "gemini-3.5-flash-lite"` (testado em `test_app.test_saude`) |
| API direta | ✅ | só `client.models.generate_content(...)` em `agente._gemini` |
| Sem function calling automático | ✅ | `AutomaticFunctionCallingConfig(disable=True)` |
| Recuperação RAG | ✅ | `recuperacao.py` (TF-IDF + cosseno, fontes, limiar); `test_recuperacao.py` |
| Consultas determinísticas | ✅ | `ferramentas.consultar_dados()` (pandas); `test_ferramentas.py` contra o CSV original |
| Modelo `.joblib` | ✅ | `modelo/modelo.joblib` (MLP, 257 KB), carrega em processo novo |
| Pré-processamento preservado | ✅ | Pipeline do notebook dentro do `.joblib`; métricas reproduzidas (etapa 03) |
| Amostra de conferência | ✅ | `modelo/amostra_conferencia.json` (6 entradas, inclui ausentes); ponte `ok: True` |
| Métricas reais | ✅ | `modelo/metricas.json` = treino real (R² 0,5537; MAE 236,03) |
| Histórico conversacional | ⚠️ | implementado e testado com Gemini simulado; reconstrução real de "E em 2024?" depende do Gemini (teste 10 pendente) |
| Etapas executadas | ✅ | campo `etapas` na API, painel na interface |
| Logs | ✅ | `app` registra pergunta truncada e etapas; `agente` registra RAG e consulta; sem a chave |
| Segurança | ✅ | limites de tamanho/taxa, XSS (textContent), histórico não confiável, consultas sem `eval`, erros controlados |
| Chave da API protegida | ✅ | só em `GEMINI_API_KEY`; `.env` no `.gitignore`; sem chave no JS/HTML/testes |
| README com "O que mudamos em relação ao kit" | ✅ | seção atualizada até a etapa 07 |
| Testes acadêmicos (≥ 10, 12 categorias) | ⏳ | bateria e gabaritos prontos (`bateria_testes.json`, `GABARITO.md`); **execução real pendente** |
| Comparação com e sem RAG | ⏳ | `comparar_sem_rag.py --executar` pronto; **não executado** |
| Declaração de uso de IA | ⚠️ | rascunho em `USO_DE_IA.md`; o grupo precisa completar as revisões humanas |
| Repositório público | ✅ | `api.github.com/repos/DanielStrapazzon/agente-rag-ml` responde 200 sem autenticação |
| Commits ao longo do trabalho pelos dois integrantes | ⚠️ ❓ | os 6 commits do grupo têm **um único autor**; o commit inicial (kit + enunciado) tem outro autor (dono do repositório remoto). O README do kit pede commits dos dois integrantes |
| Publicação da aplicação | ⏳ | plataforma não escolhida; checkpoint de publicação do kit: 26/11 |
| `links.txt` | ⏳ | ainda é o modelo do kit (grupo, app, GitHub, hash, chave reserva). Hash definitivo só na versão final |
| Chave reserva para a apresentação | ❓ | o kit recomenda uma chave de outra conta/projeto |
| Relatório final | ⏳ | etapa 08 (PDF não produzido) |
| Congelamento | ❓ | sem commits/deploy na `main` de 02/12, 23h59, até a arguição |

## Hospedagem (verificação sem deploy)

| Item | Situação |
|---|---|
| Comando | `gunicorn app:app --workers 1 --threads 4 --timeout 120` (README do kit); `gunicorn==23.0.0` no `requirements.txt` |
| Variáveis | `GEMINI_API_KEY` e `PYTHON_VERSION` (painel da hospedagem) |
| Rota de saúde | `GET /saude` → `{ok, modelo}` (não verifica o Gemini) |
| Arquivos obrigatórios na raiz | `app.py`, `agente.py`, `ferramentas.py`, `recuperacao.py`, **`previsao_layoffs.py` e `preparar_dados.py`** (o `.joblib` depende deles), `prompts/`, `conhecimento/`, `dados/dataset_legivel.csv`, `modelo/`, `static/` |
| Tamanho dos artefatos | `modelo/` 260 KB; `dados/` 516 KB — muito abaixo do limite de 512 MB do Render gratuito |
| Versão do Python | ⚠️ **divergência**: venv do treino = **3.14.10**; `.python-version` = **3.13**. O kit pede `PYTHON_VERSION` = versão completa do venv do treino |

Estratégia para manter a compatibilidade do modelo serializado (decisão do grupo, não aplicada):
1. **Preferida:** hospedar com Python **3.14.10** (`PYTHON_VERSION=3.14.10`, e alinhar `.python-version`), com as mesmas
   versões fixas de scikit-learn 1.9.1 / numpy 2.5.3 — **confirmar** se a plataforma oferece 3.14.10.
2. **Alternativa:** criar um venv 3.13, instalar o mesmo `requirements.txt`, rodar `preparar_dados.py` e `treinar.py`
   (retreino com a mesma seed e grade — exige autorização), conferir métricas/hashes e a ponte, e usar
   `PYTHON_VERSION` = essa versão 3.13.x.
Em qualquer caso, verificar no log do servidor que **não** aparece "ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino"
nem o aviso de versão do scikit-learn.
`versao_streamlit/requirements.txt` não foi testado (Streamlit não instalado no venv).
