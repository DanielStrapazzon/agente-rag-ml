"""
Proteção contra execução acidental dos scripts de EXEMPLO do kit, que gravam em dados/ e modelo/.

Desde a etapa 04, dados/dataset_legivel.csv e modelo/* são os artefatos DEFINITIVOS do Tech Layoffs.
treinar_exemplo.py, exemplo_classificacao/treinar_classificacao.py e
exemplo_transformacao/treinar_fora_do_pipeline.py chamam exigir_permissao_para_sobrescrever() antes
de gravar qualquer coisa: se os artefatos forem do Tech Layoffs, o script para com uma mensagem.

Para rodar um exemplo mesmo assim (sabendo que os artefatos serão sobrescritos):
    PERMITIR_SOBRESCREVER_ARTEFATOS=1 python treinar_exemplo.py
Para recuperar os artefatos depois:  .venv/bin/python preparar_dados.py && .venv/bin/python treinar.py
Criado na etapa 05 com apoio do Claude Code (Anthropic).
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VARIAVEL = "PERMITIR_SOBRESCREVER_ARTEFATOS"


def artefatos_tech_layoffs(raiz: Path = RAIZ) -> list[str]:
    """Arquivos em dados/ e modelo/ que pertencem ao Tech Layoffs (pelo conteúdo, não pelo nome)."""
    encontrados = []
    csv = raiz / "dados" / "dataset_legivel.csv"
    if csv.exists():
        with csv.open(encoding="utf-8", errors="ignore") as f:
            if "Laid_Off" in f.readline():
                encontrados.append("dados/dataset_legivel.csv")
    metricas = raiz / "modelo" / "metricas.json"
    if metricas.exists() and "Laid_Off" in metricas.read_text(encoding="utf-8", errors="ignore"):
        encontrados.append("modelo/metricas.json")
    return encontrados


def exigir_permissao_para_sobrescrever(script: str, raiz: Path = RAIZ) -> None:
    """Interrompe o script (SystemExit 2) se ele for sobrescrever os artefatos do Tech Layoffs
    sem a variável PERMITIR_SOBRESCREVER_ARTEFATOS=1."""
    protegidos = artefatos_tech_layoffs(raiz)
    if protegidos and os.environ.get(VARIAVEL) != "1":
        sys.stderr.write(
            f"\n[{script}] INTERROMPIDO: este é um script de EXEMPLO do kit e sobrescreveria os artefatos "
            f"definitivos do Tech Layoffs: {', '.join(protegidos)} (e modelo/modelo.joblib).\n"
            f"Se for mesmo isso que você quer, rode de novo com {VARIAVEL}=1. Para recuperar depois:\n"
            "  .venv/bin/python preparar_dados.py && .venv/bin/python treinar.py\n\n")
        raise SystemExit(2)
