"""Testa dados, modelo, consultas, RAG e verificação SEM gastar cota do Gemini:  python testar_sem_api.py
Adaptado ao Tech Layoffs na etapa 04 (com apoio do Claude Code). Os testes automatizados completos
estão em tests/ (.venv/bin/python -m unittest discover -s tests -v)."""
import json
import os

os.environ.setdefault("GEMINI_API_KEY", "teste-offline")       # o Gemini não é chamado aqui

import ferramentas as F  # noqa: E402
from agente import verificar  # noqa: E402
from recuperacao import RECUPERADOR  # noqa: E402

print("Conferência da ponte legível -> modelo:", F.conferir_transformacao())

# 0. Cenário real: 1ª linha do dataset legível (Tamara Mellon, 2020-03)
linha = F.DF.iloc[0]
exemplo = {a: (None if F._vazio(F._nativo(linha[a])) else str(F._nativo(linha[a]))) for a in F.ATRIBUTOS}
print("0. Exemplo (1ª linha do CSV):", exemplo, "| Laid_Off real:", int(linha[F.ALVO]))
print("1. Faltando (sem year e month):", F.faltando({**exemplo, "year": None, "month": ""}))
print("   Faltando (sem Continent, com Country):", F.faltando({**exemplo, "Continent": None}))
entrada, erros, alertas = F.preparar(exemplo)
print("2. Preparar:", entrada, erros, alertas)
assert not erros, f"a 1ª linha do CSV deveria ser válida: {erros}"
p = F.prever(entrada)
print("3. Previsão:", {k: p[k] for k in ("previsao", "unidade", "modelo", "erro_absoluto_medio_no_teste_MAE",
                                         "r2_teste", "previsao_inconsistente")})
c = F.casos_semelhantes(entrada, p["previsao"])
print("4. Casos semelhantes:", {k: v for k, v in c.items() if k not in ("casos", "criterio")})
print("5. Fora da faixa (alerta):", F.preparar({**exemplo, "Company_Size_before_Layoffs": "900000"})[2])
print("6. Erros (impossível/inválido):")
for caso in [{"month": "13"}, {"Company_Size_before_Layoffs": "-5"}, {"Company_Size_before_Layoffs": "12 abc"},
             {"year": "1800"}, {"Money_Raised_clean": "-1"}, {"Country": "Wakanda"}, {"Continent": "Europe"},
             {"Country": "Reino Unido"}]:
    print("  ", caso, "->", F.preparar({**exemplo, **caso})[1])
print("7. Opcional desconhecido (alerta):", F.preparar({**exemplo, "Stage": "não sei"})[2])
pequena = F.prever(F.preparar({**exemplo, "Company_Size_before_Layoffs": "6"})[0])
print("8. Previsão acima do porte:", pequena["previsao"], pequena["previsao_inconsistente"], pequena.get("aviso_inconsistencia"))

print("9. Consultas estruturadas (consultar_dados):")
for q in [{"operacao": "soma", "coluna": "Laid_Off"},
          {"operacao": "contagem"},
          {"operacao": "contagem_distinta", "coluna": "Company"},
          {"operacao": "soma", "coluna": "Laid_Off", "filtros": [{"coluna": "Country", "operador": "eq", "valor": "Brasil"}]},
          {"operacao": "ranking", "agregacao": "soma", "coluna": "Laid_Off", "agrupar_por": ["Industry"], "limite": 3},
          {"operacao": "comparacao", "agregacao": "soma", "coluna": "Laid_Off", "comparar": {"coluna": "year", "valores": [2022, 2023]}},
          {"operacao": "maximo", "coluna": "Laid_Off"},
          {"operacao": "soma", "coluna": "Percentage"}]:
    r = F.consultar_dados(q)
    json.dumps(r, allow_nan=False)
    print("  ", json.dumps(q, ensure_ascii=False), "->", r.get("erro") or (r["resultado"], r["unidade"]))
print("10. Estatísticas (contrato do kit):", F.estatisticas(F.ALVO))
print("11. RAG:")
for pergunta in ["quantas demissões houve em 2023?", "qual país teve mais demissões?", "demissões da Intel",
                 "qual setor teve mais demissões?", "qual modelo foi usado? MLP", "o que significa o MAE?",
                 "quais as limitações do modelo?", "receita de bolo"]:
    print("  ", pergunta, "->", [(t["similaridade"], t["fonte"].split("/")[-1], t["trecho"][:50])
                                 for t in RECUPERADOR.buscar(pergunta)[:2]])
print("12. Verificação OK  :", verificar(f"O MAE do modelo é {F.METRICAS['mae_teste']}".replace(".", ","), {"m": F.METRICAS}, []))
print("13. Verificação FALHA:", verificar("Foram 9.999 demitidos", {"previsao_do_modelo": p}, []))
