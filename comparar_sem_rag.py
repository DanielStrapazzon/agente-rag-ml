"""
Comparação para o relatório: as MESMAS perguntas, o MESMO modelo Gemini (agente.MODELO_GEMINI), três modos:
  (a) GEMINI PURO        — só a persona (prompts/responder.md), sem RAG, sem consultas, sem modelo, sem verificação;
  (b) AGENTE SEM RAG     — agente completo, mas com a busca textual desligada (RECUPERADOR.buscar -> []);
                           consultas determinísticas, previsão e verificação continuam ativas;
  (c) AGENTE COMPLETO    — RAG + consultas determinísticas + modelo + verificação.
(b) x (c) isola o efeito da recuperação textual; (a) x (c) mostra o efeito do conjunto (RAG +
ferramentas + verificação) — a diferença NÃO é causada só pelo RAG.

ATENÇÃO: gasta a cota do Gemini (~5 a 7 chamadas por pergunta). Por isso só roda com --executar:
    .venv/bin/python comparar_sem_rag.py              (mostra o plano, não chama a API)
    .venv/bin/python comparar_sem_rag.py --executar   (usa a chave do .env)
Adaptado ao Tech Layoffs na etapa 05 com apoio do Claude Code. Os números do gabarito vêm de conferir.py.
"""
import sys
from unittest import mock

PERGUNTAS = [
    "Quantas demissões estão registradas no dataset?",
    "Qual país registrou mais demissões?",
    "Quantas demissões ocorreram no Brasil em 2023?",
    "Qual algoritmo foi selecionado e por quê?",
    "Quais são as principais limitações do modelo?",
]


def main() -> None:
    if "--executar" not in sys.argv:
        print(__doc__)
        print("Perguntas:", *[f"  {i}. {p}" for i, p in enumerate(PERGUNTAS, 1)], sep="\n")
        print("\nNada foi enviado ao Gemini. Rode com --executar para fazer a comparação.")
        return
    from dotenv import load_dotenv
    load_dotenv()
    from google.genai import types
    import agente
    from recuperacao import RECUPERADOR

    for p in PERGUNTAS:
        puro = agente._gemini(agente.INSTRUCOES, [types.Content(role="user", parts=[types.Part(text=p)])])
        with mock.patch.object(RECUPERADOR, "buscar", return_value=[]):
            sem_rag = agente.responder(p, [])
        completo = agente.responder(p, [])
        print("=" * 80, "\nPERGUNTA:", p)
        print("\n[a) GEMINI PURO]\n", puro)
        print("\n[b) AGENTE SEM RAG]  etapas:", " · ".join(sem_rag["etapas"]), "\n", sem_rag["resposta"])
        print("\n[c) AGENTE COMPLETO]  etapas:", " · ".join(completo["etapas"]),
              "| fontes:", ", ".join(completo["fontes"]), "\n", completo["resposta"])


if __name__ == "__main__":
    main()
