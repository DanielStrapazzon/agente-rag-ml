"""
Dados e modelo do grupo: tudo o que é DETERMINÍSTICO (não usa o Gemini).

O agente (agente.py) decide QUANDO usar cada função; aqui fica O QUE cada uma faz:
  - prever(entrada)              -> roda o modelo .joblib
  - casos_semelhantes(entrada)   -> busca casos reais parecidos no dataset (validação)
  - estatisticas(coluna, grupo)  -> números exatos do dataset (pandas)

ADAPTEM ao trabalho de vocês: ALVO, NUMERICAS, CATEGORICAS, ATRIBUTOS, as regras de
_regras_do_dominio() e os textos. Retornem sempre dicts com tipos nativos (float, int,
str, list, dict): numpy.int64, NaN e DataFrame quebram a resposta em JSON.
CLASSIFICAÇÃO: usem exemplo_classificacao/ferramentas_classificacao.py como base.
"""
import json
import logging
import math
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import sklearn

log = logging.getLogger(__name__)
BASE = Path(__file__).parent
MODELO = joblib.load(BASE / "modelo" / "modelo.joblib")
METRICAS = json.loads((BASE / "modelo" / "metricas.json").read_text(encoding="utf-8"))
DF = pd.read_csv(BASE / "dados" / "dataset_legivel.csv")

TAREFA = "regressao"                                   # "regressao" ou "classificacao"
ALVO = "aluguel"                                       # coluna-alvo do SEU dataset
NUMERICAS = ["area_m2", "quartos", "vagas"]            # entradas numéricas do modelo
CATEGORICAS = ["bairro"]                               # entradas categóricas do modelo
UNIDADE_ALVO = "R$ por mês"

# Descrição de cada atributo de entrada: o agente usa isto para extrair os valores da conversa
# e para perguntar o que falta. tipo: "numero", "inteiro" ou "texto".
ATRIBUTOS = {
    "area_m2": {"tipo": "numero", "descricao": "área do imóvel em metros quadrados"},
    "quartos": {"tipo": "inteiro", "descricao": "número de quartos"},
    "vagas": {"tipo": "inteiro", "descricao": "número de vagas de garagem"},
    "bairro": {"tipo": "texto", "descricao": "bairro do imóvel"},
}

if METRICAS.get("sklearn_version") != sklearn.__version__:
    log.error("ATENÇÃO: modelo salvo com scikit-learn %s, mas o servidor usa %s. Ajustem o requirements.txt.",
              METRICAS.get("sklearn_version"), sklearn.__version__)


def _norm(texto) -> str:
    """'  Jardim Italia ' e 'jardim itália' viram a mesma chave."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).casefold().strip()


def _nativo(v):
    """numpy.int64 -> int, numpy.float64 -> float (para virar JSON)."""
    return v.item() if hasattr(v, "item") else v


CATEGORIAS = {col: {_norm(v): _nativo(v) for v in DF[col].dropna().unique()} for col in CATEGORICAS}
for col in CATEGORICAS:                                 # o agente mostra as opções válidas
    ATRIBUTOS[col]["valores"] = sorted(str(v) for v in CATEGORIAS[col].values())
FAIXAS = {col: (float(DF[col].min()), float(DF[col].max())) for col in NUMERICAS}
DESVIOS = {col: float(DF[col].std()) or 1.0 for col in NUMERICAS}
COLUNAS_CONSULTA = NUMERICAS + [ALVO]                   # colunas que estatisticas() aceita
AGRUPAMENTOS = CATEGORICAS                              # agrupamentos que estatisticas() aceita


# ---------------------------------------------------------------- PONTE: dataset legível -> modelo
# Existem DUAS versões dos dados, e as duas são necessárias:
#   - dados/dataset_legivel.csv: para PESSOAS e para o RAG (estatísticas, casos semelhantes, fatos):
#     categorias por extenso ("Centro", "feminino"), números na escala original (R$, m², anos);
#   - o formato do MODELO: o que o modelo recebeu no treino (texto -> número, escalas ajustadas...).
# para_o_modelo() é a ponte: aplica na entrada do usuário EXATAMENTE as transformações do treino.
# ESQUECER ESTA ETAPA = previsões erradas e silenciosas. Por isso, conferir_transformacao() (abaixo)
# compara, ao iniciar o app, as previsões desta ponte com as previsões salvas no treino.
def para_o_modelo(entrada: dict) -> pd.DataFrame:
    """Entrada legível (dict) -> DataFrame no formato que o modelo espera.

    Caso 1 (este kit, recomendado): o pré-processamento está DENTRO do Pipeline salvo
            (ColumnTransformer com OneHotEncoder/StandardScaler) -> basta montar o DataFrame.
    Caso 2: as transformações foram feitas FORA do Pipeline (get_dummies, LabelEncoder,
            scaler, mapa de códigos 1/2...): salvem no treino os transformadores AJUSTADOS
            e apliquem aqui, na mesma ordem. Exemplo completo em exemplo_transformacao/.
    """
    return pd.DataFrame([entrada])[NUMERICAS + CATEGORICAS]


def conferir_transformacao() -> dict:
    """Refaz as previsões da amostra salva no treino (modelo/amostra_conferencia.json) passando
    pela ponte para_o_modelo(). Se não baterem, a transformação está diferente da do treino."""
    arquivo = BASE / "modelo" / "amostra_conferencia.json"
    if not arquivo.exists():
        return {"ok": False, "detalhes": ["modelo/amostra_conferencia.json não existe: gerem no treino (veja treinar_exemplo.py)"]}
    problemas = []
    for item in json.loads(arquivo.read_text(encoding="utf-8")):
        try:
            obtido = _nativo(MODELO.predict(para_o_modelo(item["entrada"]))[0])
        except Exception as e:                     # ex.: colunas diferentes das do treino
            problemas.append(f"o modelo recusou a entrada transformada ({type(e).__name__}: {str(e)[:200]})")
            continue
        esperado = item["previsao_no_treino"]
        igual = (abs(float(obtido) - float(esperado)) <= 1e-6 * max(1.0, abs(float(esperado)))
                 if isinstance(esperado, (int, float)) and not isinstance(esperado, bool) else str(obtido) == str(esperado))
        if not igual:
            problemas.append(f"entrada {item['entrada']}: esperado {esperado}, obtido {obtido}")
    return {"ok": not problemas, "detalhes": problemas}


def faltando(entrada: dict) -> list[str]:
    """Atributos que o usuário ainda não informou."""
    return [a for a in ATRIBUTOS if entrada.get(a) in (None, "")]


def _regras_do_dominio(e: dict) -> list[str]:
    """Valores IMPOSSÍVEIS no mundo real -> erro (não prevê). ADAPTEM ao seu domínio."""
    erros = []
    if e["area_m2"] <= 0:
        erros.append("a área precisa ser maior que zero")
    if e["quartos"] < 0 or e["vagas"] < 0:
        erros.append("quartos e vagas não podem ser negativos")
    return erros


_MILHAR = r"-?[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?"


def _numero(valor) -> float:
    """'80', '80 m²', '5.000', 'R$ 5.000,00', '2,5', '5 mil' -> float (formato brasileiro)."""
    s = str(valor)
    m = re.search(_MILHAR + r"(?![\d.])|-?\d+(?:[.,]\d+)?", s)
    if not m:
        raise ValueError(valor)
    t = m.group()
    t = t.replace(".", "").replace(",", ".") if re.fullmatch(_MILHAR, t) else t.replace(",", ".")
    x = float(t)
    if not math.isfinite(x):
        raise ValueError(valor)
    return x * 1000 if re.search(r"\bmil\b", s, re.I) else x


def preparar(entrada: dict) -> tuple[dict, list[str], list[str]]:
    """Converte tipos, corrige caixa/acento e separa ERROS (impossível) de ALERTAS (fora da faixa do treino)."""
    pronto, erros, alertas = {}, [], []
    for nome, info in ATRIBUTOS.items():
        valor = entrada.get(nome)
        if info["tipo"] == "texto":
            achado = CATEGORIAS[nome].get(_norm(valor))
            if achado is None:
                erros.append(f"{nome} '{str(valor)[:40]}' não existe no dataset. Opções: {info['valores']}")
            pronto[nome] = achado
            continue
        try:
            x = _numero(valor)
        except (TypeError, ValueError):
            erros.append(f"{nome} precisa ser um número (recebido: {str(valor)[:40]!r})")
            continue
        if info["tipo"] == "inteiro" and not x.is_integer():
            erros.append(f"{nome} precisa ser um número inteiro (recebido: {str(valor)[:40]!r})")
            continue
        pronto[nome] = int(x) if info["tipo"] == "inteiro" else x
    if erros:
        return pronto, erros, alertas
    erros += _regras_do_dominio(pronto)
    for col in NUMERICAS:
        mn, mx = FAIXAS[col]
        if not erros and not (mn <= pronto[col] <= mx):
            alertas.append(f"{col}={pronto[col]:g} está fora da faixa do dataset ({mn:g} a {mx:g}); a previsão é uma extrapolação")
    return pronto, erros, alertas


def prever(entrada: dict) -> dict:
    """Roda o modelo treinado com uma entrada já preparada (use preparar() antes)."""
    valor = float(MODELO.predict(para_o_modelo(entrada))[0])
    mae = float(METRICAS["mae_teste"])
    return {                                 # contas já prontas: o Gemini não deve fazer contas
        "previsao": round(valor, 2),
        "unidade": UNIDADE_ALVO,
        "erro_medio_do_modelo_MAE": mae,
        "faixa_tipica_previsao_mais_menos_MAE": [round(valor - mae, 2), round(valor + mae, 2)],
        "r2_teste": float(METRICAS["r2_teste"]),
    }


def casos_semelhantes(entrada: dict, previsao: float | None = None, quantidade: int = 5) -> dict:
    """Casos reais mais parecidos (mesma categoria principal, distância padronizada nas numéricas)."""
    base = DF[DF[CATEGORICAS[0]] == entrada[CATEGORICAS[0]]] if CATEGORICAS else DF
    if len(base) < quantidade:
        base = DF
    dist = sum(((base[c] - entrada[c]) / DESVIOS[c]) ** 2 for c in NUMERICAS) ** 0.5
    proximos = base.assign(_d=dist).nsmallest(quantidade, "_d").drop(columns="_d")
    reais = proximos[ALVO]
    media = float(reais.mean())
    resultado = {
        "casos": json.loads(proximos.to_json(orient="records", force_ascii=False)),
        "media_real_dos_casos": round(media, 2),
        "faixa_real_dos_casos": [round(float(reais.min()), 2), round(float(reais.max()), 2)],
        "observacao": "os casos vêm do mesmo dataset usado no treino: a comparação é indicativa, não uma prova",
    }
    if previsao is not None and media:
        resultado["diferenca_vs_media"] = round(previsao - media, 2)
        resultado["diferenca_percentual_vs_media"] = round(100 * (previsao - media) / media, 1)
        resultado["previsao_dentro_da_faixa"] = bool(reais.min() <= previsao <= reais.max())
    return resultado


def estatisticas(coluna: str, agrupar_por: str = "") -> dict:
    """Estatísticas REAIS (contagem, média, mediana, mín., máx.) de uma coluna, opcionalmente por grupo."""
    if coluna not in COLUNAS_CONSULTA:
        return {"erro": f"coluna inválida; opções: {COLUNAS_CONSULTA}"}
    if agrupar_por and agrupar_por not in AGRUPAMENTOS:
        return {"erro": f"agrupamento inválido; opções: {AGRUPAMENTOS}"}
    resumo = ["count", "mean", "median", "min", "max"]
    if agrupar_por:
        tabela = DF.groupby(agrupar_por)[coluna].agg(resumo).round(2).reset_index()
        return {"coluna": coluna, "por_grupo": json.loads(tabela.to_json(orient="records", force_ascii=False))}
    serie = DF[coluna].agg(resumo).round(2)
    return {"coluna": coluna, "resumo": {k: (None if pd.isna(v) else float(v)) for k, v in serie.items()}}


def fatos_do_dataset() -> list[str]:
    """Frases com números reais, geradas do CSV, para a base de conhecimento do RAG."""
    fatos = [f"O dataset tem {len(DF)} registros. A coluna-alvo é '{ALVO}' ({UNIDADE_ALVO}).",
             f"Desempenho do modelo no teste: {METRICAS.get('modelo', '')}; R² = {METRICAS['r2_teste']}; "
             f"MAE = {METRICAS['mae_teste']} ({UNIDADE_ALVO})."]
    for col in NUMERICAS + [ALVO]:
        s = DF[col]
        desc = ATRIBUTOS[col]["descricao"] if col in ATRIBUTOS else f"alvo, em {UNIDADE_ALVO}"
        fatos.append(f"Faixa e média de '{col}' ({desc}): varia de {s.min():g} a {s.max():g}, "
                     f"com média {s.mean():.2f} e mediana {s.median():g}.")
    for col in CATEGORICAS:
        g = DF.groupby(col)[ALVO].agg(["mean", "count"]).sort_values("mean", ascending=False)
        partes = "; ".join(f"{k}: média {v['mean']:.2f} ({int(v['count'])} registros)" for k, v in g.iterrows())
        fatos.append(f"'{ALVO}' médio por '{col}', do maior para o menor: {partes}.")
    corr = DF[NUMERICAS + [ALVO]].corr()[ALVO].drop(ALVO).sort_values(key=abs, ascending=False)
    fatos.append(f"Qual variável mais influencia o {ALVO}? Correlação (r) de cada variável numérica com o {ALVO}: " +
                 "; ".join(f"{k} r = {v:.2f}" for k, v in corr.items()) + ".")
    return fatos


_conferencia = conferir_transformacao()
if not _conferencia["ok"]:
    log.error("ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino: %s", _conferencia["detalhes"][:3])
