"""
Comparação para o relatório: as MESMAS perguntas, o MESMO modelo Gemini (agente.MODELO_GEMINI), três modos:
  (a) GEMINI PURO        — só a persona (prompts/responder.md), sem RAG, sem consultas, sem modelo, sem verificação;
  (b) AGENTE SEM RAG     — agente completo, mas com a busca textual desligada (RECUPERADOR.buscar -> []);
                           consultas determinísticas, previsão e verificação continuam ativas;
  (c) AGENTE COMPLETO    — RAG + consultas determinísticas + modelo + verificação.
(b) x (c) isola o efeito da recuperação textual; (a) x (c) mostra o efeito do conjunto (RAG +
ferramentas + verificação) — a diferença NÃO é causada só pelo RAG.

ATENÇÃO: gasta a cota do Gemini (~5 chamadas por pergunta, ~25 no total). Por isso só roda com --executar:
    .venv/bin/python comparar_sem_rag.py              (mostra o plano, não chama a API)
    .venv/bin/python comparar_sem_rag.py --executar   (usa a chave do .env; grava reports/validacao/comparacao_rag_resultados.json)
    .venv/bin/python comparar_sem_rag.py --executar --continuar   (retoma: pula perguntas já registradas sem erro)
A conferência numérica usa os gabaritos de validar_agente.py (pandas no CSV original); a análise
qualitativa (completude, limitações) é feita por pessoas em reports/validacao/COMPARACAO_RAG.md.
Adaptado ao Tech Layoffs nas etapas 05 e 07 com apoio do Claude Code.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "reports" / "validacao" / "comparacao_rag_resultados.json"
PERGUNTAS = [
    "Quantas demissões estão registradas no dataset?",
    "Qual país registrou mais demissões?",
    "Quantas demissões ocorreram no Brasil em 2023?",
    "Qual algoritmo foi selecionado e por quê?",
    "Quais são as principais limitações do modelo?",
]


def gabaritos() -> list[dict]:
    """Números que uma resposta correta deve conter (pelo menos um de cada lista) e termos esperados."""
    import validar_agente as V
    g = V.calcular_gabaritos()
    pais, valor = next(iter(g["por_pais_top10_agrupando_uk"].items()))
    m = g["metricas"]
    return [
        {"numeros": [g["total_demitidos"]], "termos": []},
        {"numeros": [valor], "termos": [pais, "Estados Unidos", "EUA"]},
        {"numeros": [g["brasil_2023"]["demitidos"]], "termos": []},
        {"numeros": [m["mae_cv"], m["mae_teste"]], "termos": ["MLP"]},
        {"numeros": [], "termos": ["probabilidade", "aleatória", "assimétric", "repetid", "casos semelhantes", "extremo"]},
    ]


def conferir(resposta: str, gab: dict) -> dict:
    import validar_agente as V
    nums = V._numeros(resposta)
    return {"numero_do_gabarito_presente": (any(V._bate(n, nums) for n in gab["numeros"]) if gab["numeros"] else None),
            "termos_encontrados": [t for t in gab["termos"] if t.lower() in resposta.lower()],
            "numeros_na_resposta": nums[:20]}


def rodar(modo: str, pergunta: str, agente, types, RECUPERADOR) -> dict:
    inicio = time.monotonic()
    try:
        if modo == "gemini_puro":
            texto = agente._gemini(agente.INSTRUCOES, [types.Content(role="user", parts=[types.Part(text=pergunta)])])
            r = {"resposta": texto, "etapas": ["geração (sem RAG, sem ferramentas, sem verificação)"], "fontes": []}
        elif modo == "agente_sem_rag":
            with mock.patch.object(RECUPERADOR, "buscar", return_value=[]):
                r = agente.responder(pergunta, [])
        else:
            r = agente.responder(pergunta, [])
        r = {k: r.get(k) for k in ("resposta", "etapas", "fontes")}
    except Exception as e:                                       # erros da API são registrados, não escondidos
        codigo = getattr(e, "code", None)
        r = {"erro": {403: "erro_403", 429: "erro_429", 503: "erro_503"}.get(codigo)
             or ("timeout" if "Timeout" in type(e).__name__ else type(e).__name__), "detalhe": str(e)[:300]}
    r["segundos"] = round(time.monotonic() - inicio, 2)
    return r


def main() -> int:
    if "--executar" not in sys.argv:
        print(__doc__)
        print("Perguntas:", *[f"  {i}. {p}" for i, p in enumerate(PERGUNTAS, 1)], sep="\n")
        print("\nNada foi enviado ao Gemini. Rode com --executar para fazer a comparação.")
        return 0
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")                                    # o conteúdo do .env não é impresso
    from google.genai import types
    import agente
    from recuperacao import RECUPERADOR

    dados = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() and "--continuar" in sys.argv else \
        {"modelo": agente.MODELO_GEMINI, "inicio": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "perguntas": []}
    feitas = {p["pergunta"] for p in dados["perguntas"] if not any("erro" in r for r in p["modos"].values())}
    gabs = gabaritos()
    for pergunta, gab in zip(PERGUNTAS, gabs):
        if pergunta in feitas:
            continue
        item = {"pergunta": pergunta, "gabarito": gab, "modos": {}}
        for modo in ("gemini_puro", "agente_sem_rag", "agente_completo"):
            r = rodar(modo, pergunta, agente, types, RECUPERADOR)
            if "resposta" in r:
                r["conferencia"] = conferir(r["resposta"] or "", gab)
            item["modos"][modo] = r
            print(f"[{modo}] {pergunta[:50]:<50} {r['segundos']:>6}s " + (r.get("erro") or "ok"))
        dados["perguntas"] = [p for p in dados["perguntas"] if p["pergunta"] != pergunta] + [item]
        dados["fim"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        SAIDA.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")   # salva a cada pergunta
        if any(r.get("erro") in ("erro_429", "erro_403") for r in item["modos"].values()):
            print("Interrompido (cota/acesso). Rode depois com --executar --continuar.")
            break
    print(f"Resultados em {SAIDA.relative_to(RAIZ)}. Faça a análise qualitativa em reports/validacao/COMPARACAO_RAG.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
