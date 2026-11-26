"""
Comparação para o relatório: as MESMAS perguntas respondidas
  (a) pelo Gemini "puro", só com a persona (sem RAG, sem modelo, sem verificação), e
  (b) pelo agente completo de vocês.

    python comparar_sem_rag.py            (usa a chave do .env; gasta ~3 chamadas por pergunta)

Troquem as perguntas pelas do dataset de vocês e registrem as diferenças no relatório.
"""
from dotenv import load_dotenv

load_dotenv()
from google.genai import types  # noqa: E402

import agente  # noqa: E402

PERGUNTAS = [
    "Qual bairro tem o aluguel médio mais alto?",
    "Qual a média de aluguel do dataset?",
    "Quanto custa um apartamento de 80 m², 2 quartos e 1 vaga no Centro?",
    "Qual variável mais influencia o aluguel?",
    "Esse modelo é confiável?",
]

for p in PERGUNTAS:
    puro = agente._gemini(agente.INSTRUCOES, [types.Content(role="user", parts=[types.Part(text=p)])])
    completo = agente.responder(p, [])
    print("=" * 80, "\nPERGUNTA:", p)
    print("\n[SEM RAG]\n", puro)
    print("\n[AGENTE COMPLETO]  etapas:", " · ".join(completo["etapas"]), "\n", completo["resposta"])
