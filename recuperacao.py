"""
Recuperação (o "R" do RAG): encontra na base de conhecimento os trechos mais relacionados
com a pergunta, usando TF-IDF + similaridade de cosseno (scikit-learn, roda local e de graça).

Base de conhecimento = arquivos .md da pasta conhecimento/ (descrição do dataset, EDA, modelo e
métricas, limitações) + frases com números reais geradas do CSV (ferramentas.fatos_do_dataset()).

Melhorias em relação ao kit (etapa 04, com apoio do Claude Code):
  - cada trecho guarda a FONTE (arquivo ou "fatos do CSV") e a SEÇÃO;
  - a pergunta é EXPANDIDA com sinônimos do domínio (PT <-> EN) e com os nomes do dataset
    correspondentes aos aliases em português (Brasil -> Brazil, varejo -> Retail);
  - o resultado mantém as chaves "trecho" e "similaridade" usadas pelo agente.
"""
import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ferramentas import ALIASES, fatos_do_dataset

BASE = Path(__file__).parent
FONTE_FATOS = "fatos_do_dataset (calculados do CSV)"


def _sem_acento(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).casefold()


STOPWORDS_PT = """a o as os um uma uns umas de do da dos das no na nos nas em por para com sem sobre
e ou que se qual quais quanto quanta quantos quantas como onde quando é são foi ser ter tem há mais menos
muito pouco me te lhe isso isto esse essa este esta ao aos à às pelo pela pelos pelas meu minha seu sua
teve tiveram houve foram existe existem usado usada usados usadas sobre qual quais""".split()
STOPWORDS_PT = sorted({_sem_acento(w) for w in STOPWORDS_PT})   # igual ao texto depois do strip_accents

# Termos da pergunta (sem acento) -> termos acrescentados à busca. Só ACRESCENTA palavras.
SINONIMOS = {
    "pais": "country", "paises": "country", "continente": "continent", "setor": "industry",
    "setores": "industry", "industria": "industry", "estagio": "stage", "porte": "company_size_before_layoffs",
    "capital": "money_raised_clean", "investimento": "money_raised_clean capital", "influencia": "correlacao associada",
    "influenciam": "correlacao associada", "confiavel": "mae r2", "confianca": "confiavel mae r2",
    "erro": "mae", "precisao": "mae r2", "acuracia": "r2 acertos", "metricas": "mae rmse r2", "metrica": "mae rmse r2", "maior": "maiores", "limitacao": "limitacoes",
    "ausentes": "missing", "faltantes": "ausentes missing", "probabilidade": "chance risco", "chance": "probabilidade",
    "risco": "probabilidade", "rede": "mlp neural", "vizinhos": "knn", "periodo": "date_layoffs",
}


def _alias_para_dataset() -> list[tuple[str, str]]:
    """(alias sem acento, nomes no dataset) para os aliases em português de ferramentas.ALIASES."""
    pares = []
    for col, aliases in ALIASES.items():
        for alias, valores in aliases.items():
            pares.append((re.sub(r"[^a-z0-9]+", " ", _sem_acento(alias)).strip(), " ".join(valores)))
    return pares


ALIAS_DATASET = _alias_para_dataset()


def expandir_consulta(pergunta: str) -> str:
    """Acrescenta à pergunta os sinônimos do domínio e os nomes do dataset (ex.: 'Brasil' -> 'Brazil')."""
    texto = " " + re.sub(r"[^a-z0-9_]+", " ", _sem_acento(pergunta)) + " "
    extras = [SINONIMOS[p] for p in texto.split() if p in SINONIMOS]
    extras += [nomes for alias, nomes in ALIAS_DATASET if f" {alias} " in texto]
    return " ".join([pergunta] + extras)


def _trechos_dos_arquivos() -> list[dict]:
    """Quebra cada .md em trechos: parágrafos e itens de lista viram trechos separados,
    e cada trecho leva o título da sua seção (ex.: "Limitações: o modelo não conhece...")."""
    trechos = []
    for arquivo in sorted((BASE / "conhecimento").glob("*.md")):
        texto = arquivo.read_text(encoding="utf-8")
        titulo_doc = texto.split("\n", 1)[0].strip("# ").strip()          # o título do arquivo entra no trecho
        secao = ""
        for linha_ou_bloco in re.split(r"\n\s*\n|\n(?=\s*[-*] )|\n(?=#)", texto):
            bloco = linha_ou_bloco.strip()
            if bloco.startswith("#"):                                     # título de seção
                titulo, _, resto = bloco.partition("\n")
                secao = titulo.strip("# ").strip()
                bloco = resto.strip()
            bloco = bloco.strip(" \n-*")
            if len(bloco) > 25:
                trechos.append({"trecho": f"{titulo_doc} — {secao}: {bloco}" if secao and secao != titulo_doc else f"{titulo_doc}: {bloco}",
                                "fonte": f"conhecimento/{arquivo.name}", "secao": secao})
    return trechos


def _trechos_dos_fatos() -> list[dict]:
    return [{"trecho": f, "fonte": FONTE_FATOS, "secao": f.split(":", 1)[0]} for f in fatos_do_dataset()]


class Recuperador:
    def __init__(self):
        self.trechos = _trechos_dos_arquivos() + _trechos_dos_fatos()
        self.vetorizador = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                           stop_words=STOPWORDS_PT, ngram_range=(1, 2), sublinear_tf=True)
        self.matriz = self.vetorizador.fit_transform([t["trecho"] for t in self.trechos])

    def buscar(self, pergunta: str, k: int = 4, limiar: float = 0.08, limiar_topo: float = 0.15) -> list[dict]:
        """Até k trechos com similaridade >= limiar, do mais para o menos parecido.
        Se nem o MELHOR trecho chega a limiar_topo, a pergunta não tem relação suficiente com a base -> [].
        Cada item: {"trecho", "similaridade", "fonte", "secao"}."""
        if not pergunta or not pergunta.strip():
            return []
        notas = cosine_similarity(self.vetorizador.transform([expandir_consulta(pergunta)]), self.matriz)[0]
        ordem = notas.argsort()[::-1][:k]
        if notas[ordem[0]] < limiar_topo:
            return []
        return [{**self.trechos[i], "similaridade": round(float(notas[i]), 3)} for i in ordem if notas[i] >= limiar]


RECUPERADOR = Recuperador()
