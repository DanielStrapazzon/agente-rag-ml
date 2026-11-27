"""
Dados e modelo do Tech Layoffs: tudo o que é DETERMINÍSTICO (não usa o Gemini).

O agente (agente.py) decide QUANDO usar cada função; aqui fica O QUE cada uma faz:
  - faltando(entrada) / preparar(entrada) -> o que falta pedir; validação (erros x alertas)
  - prever(entrada)                       -> estimativa do modelo MLP (previsao_layoffs.prever_demissoes)
  - casos_semelhantes(entrada, previsao)  -> eventos reais parecidos (distância de Gower)
  - estatisticas(coluna, agrupar_por)     -> resumo de uma coluna (contrato original do kit)
  - consultar_dados(consulta)             -> consultas estruturadas: soma, contagem, ranking, comparação...
  - fatos_do_dataset()                    -> frases com números reais para o RAG

O modelo estima QUANTOS funcionários foram demitidos num evento com as características informadas.
Ele NÃO estima a probabilidade de uma empresa demitir (o dataset só tem eventos que aconteceram).

Retornos: só tipos nativos (float, int, str, list, dict, None) — JSON estrito, sem NaN/Infinity.
Adaptado do kit com apoio do Claude Code (Anthropic) na etapa 04; revisado e executado pelo grupo.
"""
import difflib
import json
import logging
import math
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from preparar_dados import ALVO, CATEGORICAS, NUMERICAS
from previsao_layoffs import (categorias_conhecidas, para_dataframe_modelo, prever_demissoes,
                              valores_imputados)

log = logging.getLogger(__name__)
BASE = Path(__file__).parent
MODELO = joblib.load(BASE / "modelo" / "modelo.joblib")
METRICAS = json.loads((BASE / "modelo" / "metricas.json").read_text(encoding="utf-8"))
DF = pd.read_csv(BASE / "dados" / "dataset_legivel.csv", low_memory=False)
DATAS = pd.to_datetime(DF["Date_layoffs"], format="%Y-%m-%d", errors="coerce")

TAREFA = "regressao"
UNIDADE_ALVO = "funcionários demitidos no evento"
OPCIONAIS = ["Stage", "Money_Raised_clean"]      # podem ser declarados "desconhecidos" (o Pipeline imputa)

# Descrição de cada atributo de entrada (nomes técnicos do dataset, descrição em português).
ATRIBUTOS = {
    "Country": {"tipo": "texto", "descricao": "país da sede da empresa (Country), nome em inglês no dataset, ex.: USA, India, Brazil"},
    "Continent": {"tipo": "texto", "descricao": "continente da sede (Continent); se faltar, é deduzido do país"},
    "Industry": {"tipo": "texto", "descricao": "setor de atuação da empresa (Industry), ex.: Finance, Retail, Healthcare"},
    "Stage": {"tipo": "texto", "descricao": "estágio de financiamento (Stage), ex.: Seed, Series B, Post-IPO, Acquired; aceita 'desconhecido'"},
    "Company_Size_before_Layoffs": {"tipo": "inteiro", "descricao": "número de funcionários da empresa antes da demissão (Company_Size_before_Layoffs)"},
    "Money_Raised_clean": {"tipo": "numero", "descricao": "capital captado pela empresa, em milhões (moeda não confirmada no dataset); aceita 'desconhecido'"},
    "year": {"tipo": "inteiro", "descricao": "ano do evento de demissão (year)"},
    "month": {"tipo": "inteiro", "descricao": "mês do evento de demissão, de 1 a 12 (month)"},
}

if METRICAS.get("sklearn_version") != sklearn.__version__:
    log.error("ATENÇÃO: modelo salvo com scikit-learn %s, mas o servidor usa %s. Ajustem o requirements.txt.",
              METRICAS.get("sklearn_version"), sklearn.__version__)


# ---------------------------------------------------------------- utilidades
def _norm(texto) -> str:
    """'  Pós-IPO ' e 'pos ipo' viram a mesma chave: sem acento, minúsculas, pontuação -> espaço."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


def _nativo(v):
    """Converte recursivamente para tipos JSON nativos; NaN/Inf/NaT -> None."""
    if isinstance(v, dict):
        return {str(k): _nativo(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nativo(x) for x in v]
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    return v


def _vazio(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


DESCONHECIDO = {_norm(t) for t in ["desconhecido", "desconhecida", "não sei", "nao sabemos", "não se sabe",
                                   "sem informação", "não divulgado", "ignorado", "indefinido", "n/a", "na",
                                   "não disponível", "unknown", "não tenho essa informação"]}


def _desconhecido(v) -> bool:
    return isinstance(v, str) and _norm(v) in DESCONHECIDO


def _fmt(x, casas: int = 0) -> str:
    """Formato brasileiro: 581.705 ; 333,36 (o verificador do agente lê esse formato)."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------- aliases e equivalências
# Aliases EXPLÍCITOS em português (e variações em inglês) -> valores ORIGINAIS do dataset.
# Um alias com mais de um valor é ambíguo: na previsão vira pedido de esclarecimento.
ALIASES = {
    "Country": {
        "Brasil": ["Brazil"], "Estados Unidos": ["USA"], "EUA": ["USA"], "Estados Unidos da America": ["USA"],
        "United States": ["USA"], "US": ["USA"], "Alemanha": ["Germany"], "Índia": ["India"],
        "Reino Unido": ["UK", "United Kingdom"], "Grã-Bretanha": ["UK", "United Kingdom"],
        "Canadá": ["Canada"], "Austrália": ["Australia"], "Áustria": ["Austria"], "Bélgica": ["Belgium"],
        "Ilhas Cayman": ["Cayman Islands"], "República Tcheca": ["Czech Republic"], "Tchéquia": ["Czech Republic"],
        "Dinamarca": ["Denmark"], "Estônia": ["Estonia"], "Finlândia": ["Finland"], "França": ["France"],
        "Indonésia": ["Indonesia"], "Irlanda": ["Ireland"], "Japão": ["Japan"], "Quênia": ["Kenya"],
        "Malásia": ["Malaysia"], "México": ["Mexico"], "Holanda": ["Netherlands"], "Países Baixos": ["Netherlands"],
        "Nova Zelândia": ["New Zealand"], "Nigéria": ["Nigeria"], "Noruega": ["Norway"], "Rússia": ["Russia"],
        "Singapura": ["Singapore"], "Coreia do Sul": ["South Korea"], "Espanha": ["Spain"], "Suécia": ["Sweden"],
        "Suíça": ["Switzerland"], "Tailândia": ["Thailand"], "Emirados Árabes Unidos": ["United Arabian Emirates"],
        "Emirados Árabes": ["United Arabian Emirates"], "United Arab Emirates": ["United Arabian Emirates"],
        "UAE": ["United Arabian Emirates"], "Uruguai": ["Uruquay"], "Uruguay": ["Uruquay"],
    },
    "Continent": {
        "América do Norte": ["North America"], "América do Sul": ["South America"], "Ásia": ["Asia"],
        "Europa": ["Europe"], "África": ["Africa"], "Oceania": ["Oceana"],
    },
    "Industry": {
        "Varejo": ["Retail"], "Saúde": ["Healthcare"], "Finanças": ["Finance"], "Financeiro": ["Finance"],
        "Serviços financeiros": ["Financial Services"], "Transporte": ["Transportation"], "Logística": ["Logistics"],
        "Alimentação": ["Food"], "Alimentos": ["Food"], "Educação": ["Education"], "Ensino superior": ["Higher Education"],
        "Imobiliário": ["Real Estate"], "Mercado imobiliário": ["Real Estate"], "Criptomoedas": ["Crypto"],
        "Cripto": ["Crypto"], "Segurança": ["Security"], "Cibersegurança": ["Cybersecurity"],
        "Segurança cibernética": ["Cybersecurity"], "Mídia": ["Media"], "Viagens": ["Travel"], "Turismo": ["Travel"],
        "Energia": ["Energy"], "Seguros": ["Insurance"], "Jurídico": ["Legal"], "Recursos humanos": ["HR"],
        "RH": ["HR"], "Infraestrutura": ["Infrastructure"], "Dados": ["Data"], "Consumo": ["Consumer"],
        "Bens de consumo": ["Consumer"], "Outros": ["Other"], "Outro": ["Other"], "Produto": ["Product"],
        "Vendas": ["Sales"], "Construção": ["Construction"], "Manufatura": ["Manufacturing"],
        "Indústria": ["Manufacturing"], "Recrutamento": ["Recruiting"], "Aeroespacial": ["Aerospace"],
        "Inteligência artificial": ["AI"], "IA": ["AI"], "Hotelaria": ["Hospitality"],
        "Publicidade": ["Advertising Services"], "Desenvolvimento de software": ["Software Development"],
        "Suporte": ["Support"], "Entretenimento": ["Entertainment Providers"], "Comércio eletrônico": ["e-commerce"],
        "E-commerce": ["e-commerce"], "Biotecnologia": ["Biotechnology Research"],
        "Consultoria": ["Business Consulting and Services"], "Veículos autônomos": ["Autonomous-driving vehicles"],
        "Montadora": ["Motor Vehicle Manufacturing"], "Jogos": ["Gaming", "Computer Games"],
        "Games": ["Gaming", "Computer Games"], "Semicondutores": ["Semiconductors", "Semiconductor Manufacturing"],
        "Serviços de TI": ["IT Services", "IT Services and IT Consulting"],
        "Eletrodomésticos": ["Appliances"], "Hardware": ["Hardware"],
    },
    "Stage": {
        "Semente": ["Seed"], "Pós-IPO": ["Post-IPO"], "Capital aberto": ["Post-IPO"], "Listada em bolsa": ["Post-IPO"],
        "Adquirida": ["Acquired"], "Adquirido": ["Acquired"], "Subsidiária": ["Subsidiary"],
        **{f"Série {letra}": [f"Series {letra}"] for letra in "ABCDEFGHIJ"},
    },
}
# Grafias do MESMO valor no dataset, agrupadas só nas CONSULTAS (o modelo as trata como categorias distintas).
EQUIVALENTES_CONSULTA = {
    "Country": [["UK", "United Kingdom"]],
    "Industry": [["Transportation", "Transportion"], ["Logistics", "Logistic"],
                 ["Higher Education", "Higher Education Industry"]],
}

CATEGORIAS = {col: sorted(str(v) for v in DF[col].dropna().unique()) for col in CATEGORICAS}
for col in CATEGORICAS:                                 # o agente mostra as opções válidas
    ATRIBUTOS[col]["valores"] = CATEGORIAS[col]
CATEGORIAS_DO_MODELO = categorias_conhecidas(MODELO)
IMPUTADOS = valores_imputados(MODELO)
CONTINENTE_DO_PAIS = {p: g.dropna().unique().tolist() for p, g in DF.groupby("Country")["Continent"]}
FAIXAS = {col: (float(DF[col].min()), float(DF[col].max())) for col in NUMERICAS}
LIMITES_ANO = (1900, 2100)                              # fora disso: ano inválido (erro)


def _indice(col: str) -> dict[str, list[str]]:
    """chave normalizada -> valores originais do dataset (inclui aliases)."""
    indice: dict[str, list[str]] = {}
    for v in DF[col].dropna().astype(str).unique():
        indice.setdefault(_norm(v), []).append(v)
    for alias, valores in ALIASES.get(col, {}).items():
        existentes = [v for v in valores if v in set(DF[col].dropna().astype(str))]
        if existentes:
            indice.setdefault(_norm(alias), [])
            indice[_norm(alias)] = sorted(set(indice[_norm(alias)]) | set(existentes))
    return indice


INDICES = {col: _indice(col) for col in ["Company", "Region"] + CATEGORICAS}


def _expandir_equivalentes(col: str, valores: list[str]) -> list[str]:
    saida = set(valores)
    for grupo in EQUIVALENTES_CONSULTA.get(col, []):
        if saida & set(grupo):
            saida |= set(grupo)
    return sorted(saida)


def resolver_categoria(col: str, valor, modo: str = "consulta") -> dict:
    """Traduz o texto do usuário para valores ORIGINAIS do dataset.
    modo="consulta": devolve todos os valores equivalentes (agrupa grafias e aliases).
    modo="previsao": exige um único valor; se ambíguo, devolve erro com as opções.
    Retorno: {"valores": [...], "erro": str|None, "observacao": str|None}."""
    if col not in INDICES:
        return {"valores": [], "erro": f"coluna '{col}' não aceita filtro por texto", "observacao": None}
    chave = _norm(valor)
    valores = list(INDICES[col].get(chave, []))
    if not valores:
        sugestoes = difflib.get_close_matches(chave, list(INDICES[col]), n=5, cutoff=0.75)
        nomes = sorted({v for s in sugestoes for v in INDICES[col][s]})
        dica = f" Talvez: {nomes}." if nomes else ""
        return {"valores": [], "erro": f"{col} '{str(valor)[:60]}' não existe no dataset.{dica}", "observacao": None}
    if modo == "previsao":
        if len(valores) > 1:
            return {"valores": valores, "erro": f"{col} '{valor}' é ambíguo no dataset; escolha uma opção: {valores}",
                    "observacao": None}
        return {"valores": valores, "erro": None, "observacao": None}
    valores = _expandir_equivalentes(col, valores)
    obs = None
    if len(valores) > 1:
        obs = f"{col} '{valor}' agrupa as grafias registradas no dataset: {valores}"
    return {"valores": valores, "erro": None, "observacao": obs}


# ---------------------------------------------------------------- conversão numérica estrita
_NUM = r"[+-]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
_MESES = {m: i for i, nomes in enumerate([
    ["janeiro", "jan", "january"], ["fevereiro", "fev", "february", "feb"], ["marco", "mar", "march"],
    ["abril", "abr", "april", "apr"], ["maio", "mai", "may"], ["junho", "jun", "june"],
    ["julho", "jul", "july"], ["agosto", "ago", "august", "aug"], ["setembro", "set", "september", "sep"],
    ["outubro", "out", "october", "oct"], ["novembro", "nov", "november"], ["dezembro", "dez", "december", "dec"]],
    start=1) for m in nomes}


def _sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


def _para_float(t: str) -> float:
    if re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):        # 5.000 / 2.961,28
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", ".")
    return float(t)


def converter_numero(col: str, valor) -> tuple[float | None, str | None, list[str]]:
    """Texto/número do usuário -> número, com casamento COMPLETO (sem aceitar '12 abc').
    Retorna (valor, erro, alertas). Unidades aceitas por coluna:
      Company_Size_before_Layoffs: '5000', '5.000', '5 mil', '5 mil funcionários'
      Money_Raised_clean (milhões): '300', '300 milhões', '1,2 bilhão' (=1200); 'mil' é recusado
      month: 1..12 ou nome ('abril', 'apr'); year: '2023'."""
    if isinstance(valor, bool):
        return None, f"{col} precisa ser numérico (recebido: {valor!r})", []
    if isinstance(valor, (int, float, np.integer, np.floating)):
        x = float(valor)
        if not math.isfinite(x):
            return None, f"{col} precisa ser um número finito (recebido: {valor!r})", []
        return x, None, []
    s = _sem_acento(str(valor))
    alertas = []
    if col == "month" and s in _MESES:
        return float(_MESES[s]), None, []
    if col == "Money_Raised_clean":
        m = re.fullmatch(r"(us\$|r\$|\$|usd|brl|eur|€)?\s*(" + _NUM + r")\s*"
                         r"(milhoes|milhao|mi|mm|million|millions|bilhoes|bilhao|bi|billion|billions|mil)?", s)
        if not m:
            return None, f"{col} precisa ser um número em milhões (recebido: {str(valor)[:40]!r})", []
        moeda, numero, unidade = m.groups()
        if unidade == "mil":
            return None, f"{col} deve ser informado em milhões (ex.: 0,5 para meio milhão); recebido: {str(valor)[:40]!r}", []
        x = _para_float(numero) * (1000 if unidade and unidade.startswith(("bilh", "bi")) else 1)
        if moeda:
            alertas.append("símbolo de moeda ignorado: o dataset registra o capital em milhões, sem moeda confirmada")
        return x, None, alertas
    sufixo = r"(mil)?\s*(funcionarios|empregados|pessoas|colaboradores)?" if col == "Company_Size_before_Layoffs" else ""
    m = re.fullmatch(r"(" + _NUM + r")\s*" + sufixo, s)
    if not m:
        return None, f"{col} precisa ser numérico (recebido: {str(valor)[:40]!r})", []
    x = _para_float(m.group(1))
    if col == "Company_Size_before_Layoffs" and m.group(2):
        x *= 1000
    return x, None, alertas


# ---------------------------------------------------------------- PONTE: dataset legível -> modelo
def para_o_modelo(entrada: dict) -> pd.DataFrame:
    """Entrada legível -> DataFrame do treino. Delegado a previsao_layoffs.para_dataframe_modelo():
    ordem das 8 colunas, categóricas object (ausente = np.nan), numéricas float64. Imputação, escala
    e one-hot estão DENTRO do modelo salvo (Pipeline) — nada é ajustado aqui."""
    return para_dataframe_modelo(entrada)


def conferir_transformacao() -> dict:
    """Refaz as previsões da amostra salva no treino (modelo/amostra_conferencia.json) passando
    pela ponte para_o_modelo(). Se não baterem, a transformação está diferente da do treino."""
    arquivo = BASE / "modelo" / "amostra_conferencia.json"
    if not arquivo.exists():
        return {"ok": False, "detalhes": ["modelo/amostra_conferencia.json não existe: gerem no treino (treinar.py)"]}
    problemas = []
    for item in json.loads(arquivo.read_text(encoding="utf-8")):
        try:
            obtido = _nativo(MODELO.predict(para_o_modelo(item["entrada"]))[0])
        except Exception as e:                     # ex.: colunas diferentes das do treino
            problemas.append(f"o modelo recusou a entrada transformada ({type(e).__name__}: {str(e)[:200]})")
            continue
        esperado = item["previsao_no_treino"]
        if abs(float(obtido) - float(esperado)) > 1e-6 * max(1.0, abs(float(esperado))):
            problemas.append(f"entrada {item['entrada']}: esperado {esperado}, obtido {obtido}")
    return {"ok": not problemas, "detalhes": problemas}


# ---------------------------------------------------------------- validação da entrada
def faltando(entrada: dict) -> list[str]:
    """Atributos que o usuário ainda NÃO respondeu (None/""). 'desconhecido' conta como resposta.
    Continent não é pedido quando o país foi informado: ele é deduzido do país."""
    falta = []
    for a in ATRIBUTOS:
        if _vazio(entrada.get(a)):
            if a == "Continent" and not _vazio(entrada.get("Country")):
                continue
            falta.append(a)
    return falta


def preparar(entrada: dict) -> tuple[dict, list[str], list[str]]:
    """Converte e valida. Separa:
      ERROS   (não prevê): obrigatório não informado ou declarado desconhecido, valor inválido, categoria
               inexistente/ambígua, valor impossível (porte <= 0, capital < 0, mês fora de 1..12, ano inválido),
               país e continente incompatíveis;
      ALERTAS (prevê com aviso): opcional desconhecido (imputado pelo Pipeline), continente deduzido,
               fora da faixa observada (extrapolação), categoria fora do treino."""
    pronto, erros, alertas = {}, [], []
    for col in CATEGORICAS:
        v = entrada.get(col)
        if _vazio(v):
            if col != "Continent":
                erros.append(f"{col} não informado")
            pronto[col] = None
            continue
        r = resolver_categoria(col, v, "previsao")
        if r["erro"] and _desconhecido(v):
            if col in OPCIONAIS:
                pronto[col] = None
                extra = " (o dataset também tem a categoria 'Unknown', que pode ser informada explicitamente)" if col == "Stage" else ""
                alertas.append(f"{col} declarado desconhecido: o Pipeline usa o valor mais frequente do treino "
                               f"('{IMPUTADOS[col]}'){extra}")
            else:
                erros.append(f"{col} é obrigatório para a estimativa e não pode ser desconhecido")
                pronto[col] = None
            continue
        if r["erro"]:
            erros.append(r["erro"])
            pronto[col] = None
            continue
        pronto[col] = r["valores"][0]
        if pronto[col] not in CATEGORIAS_DO_MODELO[col]:
            alertas.append(f"{col} '{pronto[col]}' existe no dataset, mas não apareceu no treino: o modelo a ignora")
    # continente: deduzido do país, ou conferido com ele
    pais, cont = pronto.get("Country"), pronto.get("Continent")
    if pais and len(CONTINENTE_DO_PAIS.get(pais, [])) == 1:
        esperado = CONTINENTE_DO_PAIS[pais][0]
        if cont is None and _vazio(entrada.get("Continent")):
            pronto["Continent"] = esperado
            alertas.append(f"Continent deduzido do país: {pais} -> {esperado}")
        elif cont is not None and cont != esperado:
            erros.append(f"país e continente incompatíveis: no dataset, {pais} pertence a {esperado}, não a {cont}")
    for col in NUMERICAS:
        v = entrada.get(col)
        if _vazio(v):
            erros.append(f"{col} não informado")
            pronto[col] = None
            continue
        if _desconhecido(v):
            if col in OPCIONAIS:
                pronto[col] = None
                alertas.append(f"{col} declarado desconhecido: o Pipeline usa a mediana do treino ({_fmt(IMPUTADOS[col], 1)} milhões)")
            else:
                erros.append(f"{col} é obrigatório para a estimativa e não pode ser desconhecido")
                pronto[col] = None
            continue
        x, erro, avisos = converter_numero(col, v)
        alertas += avisos
        if erro:
            erros.append(erro)
            pronto[col] = None
            continue
        inteiro = ATRIBUTOS[col]["tipo"] == "inteiro"
        if inteiro and not float(x).is_integer():
            erros.append(f"{col} precisa ser um número inteiro (recebido: {str(v)[:40]!r})")
            pronto[col] = None
            continue
        pronto[col] = int(x) if inteiro else float(x)
    erros += _regras_do_dominio(pronto)
    if not erros:
        for col in NUMERICAS:
            mn, mx = FAIXAS[col]
            if pronto[col] is not None and not (mn <= pronto[col] <= mx):
                alertas.append(f"{col}={pronto[col]:g} está fora da faixa observada no dataset ({mn:g} a {mx:g}); "
                               "a previsão é uma extrapolação")
    return {a: pronto.get(a) for a in ATRIBUTOS}, erros, alertas


TIPOS_ALERTA = {                      # trecho da mensagem -> tipo (usado pelo agente para rotular as etapas)
    "fora da faixa observada": "extrapolacao",
    "declarado desconhecido": "imputacao",
    "deduzido do país": "deducao",
    "não apareceu no treino": "categoria_fora_do_treino",
    "símbolo de moeda": "conversao",
}


def classificar_alertas(alertas: list[str]) -> dict[str, list[str]]:
    """Agrupa os alertas de preparar() por tipo: extrapolacao, imputacao, deducao,
    categoria_fora_do_treino, conversao, outros. Só tipos com alertas aparecem."""
    grupos: dict[str, list[str]] = {}
    for a in alertas:
        tipo = next((t for trecho, t in TIPOS_ALERTA.items() if trecho in a), "outros")
        grupos.setdefault(tipo, []).append(a)
    return grupos


def _regras_do_dominio(e: dict) -> list[str]:
    """Valores IMPOSSÍVEIS -> erro (não prevê)."""
    erros = []
    if e.get("Company_Size_before_Layoffs") is not None and e["Company_Size_before_Layoffs"] <= 0:
        erros.append("o número de funcionários antes da demissão precisa ser maior que zero")
    if e.get("Money_Raised_clean") is not None and e["Money_Raised_clean"] < 0:
        erros.append("o capital captado não pode ser negativo")
    if e.get("month") is not None and not 1 <= e["month"] <= 12:
        erros.append("o mês precisa estar entre 1 e 12")
    if e.get("year") is not None and not LIMITES_ANO[0] <= e["year"] <= LIMITES_ANO[1]:
        erros.append(f"ano inválido: {e['year']}")
    return erros


# ---------------------------------------------------------------- previsão
def prever(entrada: dict) -> dict:
    """Roda o modelo carregado UMA vez (MODELO) com uma entrada já preparada (use preparar() antes)."""
    r = prever_demissoes(entrada, MODELO)
    valor = r["previsao"]
    mae = float(METRICAS["mae_teste"])
    porte = entrada.get("Company_Size_before_Layoffs")
    inconsistente = porte is not None and valor > porte
    saida = {                                # contas já prontas: o Gemini não deve fazer contas
        "previsao": round(valor, 2),
        "unidade": UNIDADE_ALVO,
        "o_que_e": "estimativa do número de funcionários demitidos num evento com este cenário; "
                   "não é a probabilidade de a empresa demitir",
        "modelo": METRICAS["modelo"],
        "erro_absoluto_medio_no_teste_MAE": mae,
        "rmse_teste": float(METRICAS["rmse_teste"]),
        "r2_teste": float(METRICAS["r2_teste"]),
        "faixa_ilustrativa_previsao_mais_menos_MAE": [round(max(valor - mae, 0.0), 2), round(valor + mae, 2)],
        "aviso_sobre_a_faixa": "faixa heurística (previsão ± MAE, limitada a 0): NÃO é intervalo de confiança e "
                               "não tem garantia probabilística; o MAE é o erro absoluto médio observado no teste",
        "alertas_do_modelo": [a for a in r["alertas"] if "não informado" not in a and "não existe no treino" not in a],
        "previsao_inconsistente": bool(inconsistente),
    }
    if inconsistente:
        saida["aviso_inconsistencia"] = (f"a estimativa ({_fmt(valor, 1)}) supera o número de funcionários informado "
                                         f"({_fmt(porte)}): não é uma quantidade plausível de demissões. O valor original "
                                         "foi mantido para conferência.")
    return _nativo(saida)


# ---------------------------------------------------------------- casos semelhantes (Gower)
_LOG = ["Company_Size_before_Layoffs", "Money_Raised_clean"]     # muito assimétricas: comparadas em log1p
LIMIAR_SEMELHANTE = 0.20     # calibrado: 78% dos cenários do teste têm 5 eventos a <= 0,20 (0,15 -> 54%)


def _num_para_distancia(col, serie_ou_valor):
    return np.log1p(serie_ou_valor) if col in _LOG else serie_ou_valor


_AMPLITUDES = {}
for _c in NUMERICAS:
    _s = _num_para_distancia(_c, DF[_c].astype(float)).dropna()
    _AMPLITUDES[_c] = float(_s.max() - _s.min()) or 1.0


def distancia_gower(entrada: dict) -> pd.Series:
    """Distância de Gower de cada evento do dataset até a entrada (0 = idêntico, 1 = máximo).
    Categóricas: 0 se igual, 1 se diferente. Numéricas: |a-b| / amplitude do dataset (porte e capital
    em log1p). Atributo ausente na entrada OU no evento é ignorado e os pesos são renormalizados."""
    partes = []
    for col in CATEGORICAS:
        v = entrada.get(col)
        if _vazio(v):
            continue
        partes.append((DF[col] != v).astype(float).where(DF[col].notna()))
    for col in NUMERICAS:
        v = entrada.get(col)
        if _vazio(v):
            continue
        a = _num_para_distancia(col, DF[col].astype(float))
        b = float(_num_para_distancia(col, float(v)))
        partes.append(((a - b).abs() / _AMPLITUDES[col]).clip(upper=1.0))
    if not partes:
        return pd.Series(np.nan, index=DF.index)
    return pd.concat(partes, axis=1).mean(axis=1, skipna=True)


def casos_semelhantes(entrada: dict, previsao: float | None = None, quantidade: int = 5) -> dict:
    """Eventos reais mais parecidos com o cenário (distância de Gower sobre as 8 entradas do modelo)."""
    dist = distancia_gower(entrada).dropna()
    proximos_idx = dist.nsmallest(quantidade).index
    proximos = DF.loc[proximos_idx]
    n_proximos = int((dist.loc[proximos_idx] <= LIMIAR_SEMELHANTE).sum())
    reais = proximos[ALVO].astype(float)
    colunas = ["Company", "Date_layoffs", "Country", "Industry", "Stage", "Company_Size_before_Layoffs",
               "Money_Raised_clean", "year", "month", ALVO]
    casos = [{**{c: _nativo(linha[c]) for c in colunas}, "distancia": round(float(dist.loc[i]), 3)}
             for i, linha in proximos[colunas].iterrows()]
    resultado = {
        "criterio": "distância de Gower nas 8 entradas do modelo (categóricas: igual/diferente; porte e capital "
                    f"em log; ausentes ignorados); 'próximo' = distância <= {LIMIAR_SEMELHANTE}",
        "quantidade_casos": len(casos),
        "casos_proximos": n_proximos,
        "fallback": n_proximos < len(casos),
        "casos": casos,
        "media_real_dos_casos": round(float(reais.mean()), 2) if len(reais) else None,
        "mediana_real_dos_casos": round(float(reais.median()), 2) if len(reais) else None,
        "faixa_real_dos_casos": [round(float(reais.min()), 2), round(float(reais.max()), 2)] if len(reais) else None,
        "observacao": "os casos vêm do mesmo dataset usado para desenvolver o modelo: a comparação é indicativa, "
                      "não uma validação independente",
    }
    if resultado["fallback"]:
        resultado["aviso_fallback"] = (f"só {n_proximos} de {len(casos)} casos estão a distância <= {LIMIAR_SEMELHANTE}; "
                                       "os demais são os mais próximos disponíveis, mas menos parecidos")
    if previsao is not None and len(reais):
        media = float(reais.mean())
        resultado["diferenca_vs_media"] = round(previsao - media, 2)
        resultado["diferenca_percentual_vs_media"] = round(100 * (previsao - media) / media, 1) if media else None
        resultado["previsao_dentro_da_faixa"] = bool(reais.min() <= previsao <= reais.max())
    return _nativo(resultado)


# ---------------------------------------------------------------- consultas estatísticas
COLUNAS_CONSULTA = [ALVO, "Company_Size_before_Layoffs", "Company_Size_after_layoffs", "Percentage", "Money_Raised_clean"]
AGRUPAMENTOS = ["Country", "Continent", "Industry", "Stage", "Region", "year", "month", "Company"]
UNIDADES = {ALVO: "funcionários demitidos", "Company_Size_before_Layoffs": "funcionários (porte antes)",
            "Company_Size_after_layoffs": "funcionários (porte depois)", "Percentage": "% do quadro demitido",
            "Money_Raised_clean": "milhões (moeda não confirmada no dataset)"}
TIPOS_FILTRO = {"Company": "texto", "Country": "texto", "Continent": "texto", "Industry": "texto",
                "Stage": "texto", "Region": "texto", "year": "inteiro", "month": "inteiro", "Date_layoffs": "data",
                ALVO: "numero", "Company_Size_before_Layoffs": "numero", "Company_Size_after_layoffs": "numero",
                "Percentage": "numero", "Money_Raised_clean": "numero"}
OPERADORES = {"texto": {"eq", "ne", "in", "not_in"},
              "inteiro": {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between"},
              "numero": {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between"},
              "data": {"eq", "gt", "gte", "lt", "lte", "between"}}
AGREGACOES = {"soma": "sum", "contagem": "size", "contagem_distinta": "nunique", "media": "mean",
              "mediana": "median", "minimo": "min", "maximo": "max", "desvio_padrao": "std"}
OPERACOES = sorted(AGREGACOES) + ["ranking", "comparacao", "correlacao"]
SOMAVEIS = [ALVO]          # somar porte, capital ou percentual de vários eventos não tem significado
LIMITE_MAXIMO = 50
_DIVERGE_ANO = (DATAS.dt.year != DF["year"]).fillna(False)
_AUSENCIA = ("nenhum registro no dataset atende aos filtros; isso não significa que não houve demissões "
             "no mundo real, apenas que não há eventos registrados nesta base")


class ConsultaInvalida(ValueError):
    pass


def _valor_filtro(col: str, tipo: str, valor):
    """Converte UM valor de filtro; devolve (valores_originais | número | data, observação)."""
    if not isinstance(valor, (str, int, float)) or isinstance(valor, bool):
        raise ConsultaInvalida(f"valor de filtro inválido em {col}: use texto ou número (recebido: {type(valor).__name__})")
    if tipo == "texto":
        r = resolver_categoria(col, valor, "consulta")
        if r["erro"]:
            raise ConsultaInvalida(r["erro"])
        return r["valores"], r["observacao"]
    if tipo == "data":
        if not isinstance(valor, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", valor.strip()):
            raise ConsultaInvalida(f"data inválida em {col}: use AAAA-MM-DD (recebido: {str(valor)[:30]!r})")
        try:
            return pd.Timestamp(valor.strip()), None
        except ValueError:
            raise ConsultaInvalida(f"data inexistente em {col}: {valor!r}") from None
    x, erro, _ = converter_numero(col, valor)
    if erro:
        raise ConsultaInvalida(erro)
    if tipo == "inteiro" and not float(x).is_integer():
        raise ConsultaInvalida(f"{col} precisa ser inteiro (recebido: {valor!r})")
    return (int(x) if tipo == "inteiro" else x), None


def _aplicar_filtros(filtros) -> tuple[pd.Series, list[dict], list[str]]:
    if filtros is None:
        filtros = []
    if not isinstance(filtros, list) or len(filtros) > 10:
        raise ConsultaInvalida("filtros deve ser uma lista com no máximo 10 itens")
    mascara = pd.Series(True, index=DF.index)
    aplicados, obs = [], []
    for f in filtros:
        if not isinstance(f, dict):
            raise ConsultaInvalida("cada filtro deve ser um objeto {coluna, operador, valor}")
        col, op, valor = f.get("coluna"), f.get("operador", "eq"), f.get("valor")
        if col not in TIPOS_FILTRO:
            raise ConsultaInvalida(f"coluna de filtro não permitida: {col!r}; opções: {sorted(TIPOS_FILTRO)}")
        tipo = TIPOS_FILTRO[col]
        if op not in OPERADORES[tipo]:
            raise ConsultaInvalida(f"operador {op!r} não permitido para {col}; opções: {sorted(OPERADORES[tipo])}")
        serie = DATAS if tipo == "data" else DF[col]
        if op in ("in", "not_in", "between"):
            if not isinstance(valor, list) or not valor or (op == "between" and len(valor) != 2):
                raise ConsultaInvalida(f"o operador {op} exige uma lista" + (" com 2 valores" if op == "between" else ""))
            convertidos = [_valor_filtro(col, tipo, v) for v in valor]
        else:
            if isinstance(valor, (list, dict)):
                raise ConsultaInvalida(f"o operador {op} exige um único valor")
            convertidos = [_valor_filtro(col, tipo, valor)]
        obs += [o for _, o in convertidos if o]
        if tipo == "texto":
            alvo = sorted({x for v, _ in convertidos for x in v})
            if col == "Company":                       # mesma empresa com caixa diferente
                alvo = sorted({x for v in alvo for x in INDICES["Company"][_norm(v)]})
            m = DF[col].isin(alvo)
            mascara &= ~m if op in ("ne", "not_in") else m
            aplicados.append({"coluna": col, "operador": op, "valor": alvo})
            continue
        vals = [v for v, _ in convertidos]
        if op == "eq":
            m = serie == vals[0]
        elif op == "ne":
            m = serie != vals[0]
        elif op == "in":
            m = serie.isin(vals)
        elif op == "not_in":
            m = ~serie.isin(vals)
        elif op == "between":
            m = serie.between(min(vals), max(vals))
        else:
            m = {"gt": serie > vals[0], "gte": serie >= vals[0], "lt": serie < vals[0], "lte": serie <= vals[0]}[op]
        mascara &= m.fillna(False).astype(bool)
        aplicados.append({"coluna": col, "operador": op, "valor": _nativo(vals if len(vals) > 1 else vals[0])})
    usa_year = any(a["coluna"] == "year" for a in aplicados)
    usa_data = any(a["coluna"] == "Date_layoffs" for a in aplicados)
    if usa_data:
        obs.append("período filtrado pela data exata do evento (Date_layoffs)")
    if usa_year and usa_data:
        obs.append("atenção: a consulta combina year (coluna Year) e Date_layoffs, que divergem em 7 eventos do dataset")
    return mascara, aplicados, obs


def _chave_grupo(col: str, serie: pd.Series) -> tuple[pd.Series, list[str]]:
    """Rótulo de grupo agrupando grafias equivalentes (UK/United Kingdom; empresas com caixa diferente)."""
    obs = []
    if col in ("year", "month"):
        return serie, obs
    rotulo = serie.astype(object).where(serie.notna(), "(não informado)")
    grupos = [list(g) for g in EQUIVALENTES_CONSULTA.get(col, [])]
    if col == "Company":
        grupos = [v for v in INDICES["Company"].values() if len(v) > 1]
    for g in grupos:
        presentes = [v for v in g if (serie == v).any()]
        if len(presentes) > 1:
            canonico = max(presentes, key=lambda v: int((serie == v).sum()))
            rotulo = rotulo.where(~serie.isin(presentes), canonico)
            obs.append(f"{col}: grafias {presentes} agrupadas como '{canonico}'")
    return rotulo, obs


def _agregar(serie: pd.Series, agregacao: str):
    if agregacao == "contagem":
        return int(len(serie))
    if agregacao == "contagem_distinta":
        return int(serie.dropna().nunique())
    valores = serie.dropna().astype(float)
    if valores.empty or (agregacao == "desvio_padrao" and len(valores) < 2):
        return None
    return round(float(getattr(valores, AGREGACOES[agregacao])()), 2)


def _validar_coluna(agregacao: str, coluna):
    if agregacao == "contagem":
        return None
    if agregacao == "contagem_distinta":
        coluna = coluna or "Company"
        if coluna not in AGRUPAMENTOS:
            raise ConsultaInvalida(f"contagem_distinta aceita: {AGRUPAMENTOS}")
        return coluna
    coluna = coluna or ALVO
    if coluna not in COLUNAS_CONSULTA:
        raise ConsultaInvalida(f"coluna inválida: {coluna!r}; opções: {COLUNAS_CONSULTA}")
    if agregacao == "soma" and coluna not in SOMAVEIS:
        raise ConsultaInvalida(f"soma só é permitida em {SOMAVEIS} (somar {coluna} de vários eventos não tem significado)")
    return coluna


def _unidade(agregacao: str, coluna) -> str:
    if agregacao == "contagem":
        return "eventos de demissão"
    if agregacao == "contagem_distinta":
        return {"Company": "empresas distintas", "Country": "países distintos"}.get(coluna, f"valores distintos de {coluna}")
    base = UNIDADES.get(coluna, "")
    return f"{base} por evento" if agregacao in ("media", "mediana", "desvio_padrao") and coluna == ALVO else base


def _extremos(sub: pd.DataFrame, coluna: str, valor) -> list[dict]:
    linhas = sub[sub[coluna].astype(float) == valor].head(5)
    return [_nativo({"Company": r["Company"], "Date_layoffs": r["Date_layoffs"], "Country": r["Country"],
                     "Industry": r["Industry"], coluna: r[coluna]}) for _, r in linhas.iterrows()]


def consultar_dados(consulta: dict) -> dict:
    """Consulta estatística ESTRUTURADA sobre dados/dataset_legivel.csv (somente pandas e listas permitidas).

    {"operacao": "soma" | "contagem" | "contagem_distinta" | "media" | "mediana" | "minimo" | "maximo" |
                 "desvio_padrao" | "ranking" | "comparacao" | "correlacao",
     "coluna": "Laid_Off",                     # coluna numérica (ou a coluna da contagem_distinta)
     "filtros": [{"coluna": "Country", "operador": "eq", "valor": "Brasil"}, ...],
     "agrupar_por": ["year"],                  # até 2 colunas de AGRUPAMENTOS
     "ordem": "desc" | "asc", "limite": 10,    # para resultados agrupados / ranking
     "agregacao": "soma",                      # ranking e comparacao
     "comparar": {"coluna": "year", "valores": [2022, 2023]},   # comparacao
     "colunas": ["Laid_Off", "Company_Size_before_Layoffs"], "metodo": "pearson" | "spearman"}  # correlacao

    contagem = número de EVENTOS; contagem_distinta de Company = número de EMPRESAS;
    soma de Laid_Off = número de FUNCIONÁRIOS demitidos.
    Retorno: operacao, coluna, unidade, filtros_aplicados, registros_encontrados, agrupar_por,
    resultado, observacoes — ou {"erro": ...}."""
    try:
        return _nativo(_consultar(consulta))
    except ConsultaInvalida as e:
        return {"erro": str(e), "operacoes_permitidas": OPERACOES}
    except (TypeError, ValueError, KeyError, AttributeError) as e:     # estrutura inesperada: erro controlado
        log.warning("consulta malformada rejeitada: %s", type(e).__name__)
        return {"erro": "consulta malformada: verifique os tipos dos campos", "operacoes_permitidas": OPERACOES}


def _consultar(consulta: dict) -> dict:
    if not isinstance(consulta, dict):
        raise ConsultaInvalida("a consulta deve ser um objeto JSON")
    op = consulta.get("operacao")
    if op not in OPERACOES:
        raise ConsultaInvalida(f"operação não permitida: {op!r}")
    mascara, aplicados, obs = _aplicar_filtros(consulta.get("filtros"))
    sub = DF[mascara]
    if any(a["coluna"] == "year" for a in aplicados) or "year" in (consulta.get("agrupar_por") or []) or \
            (consulta.get("comparar") or {}).get("coluna") == "year":
        n = int(_DIVERGE_ANO[mascara].sum())
        if n:
            obs.append(f"{n} evento(s) deste resultado têm year (coluna Year) diferente do ano de Date_layoffs; "
                       "usamos year, como no trabalho anterior")
    base = {"operacao": op, "filtros_aplicados": aplicados, "registros_encontrados": int(len(sub))}

    if op == "correlacao":
        colunas, metodo = consulta.get("colunas") or [], consulta.get("metodo", "pearson")
        validas = COLUNAS_CONSULTA + ["year", "month"]
        if len(colunas) != 2 or any(c not in validas for c in colunas) or metodo not in ("pearson", "spearman"):
            raise ConsultaInvalida(f"correlacao exige 'colunas' com 2 itens de {validas} e metodo pearson|spearman")
        pares = sub[colunas].astype(float).dropna()
        valor = None
        if len(pares) >= 3 and pares[colunas[0]].nunique() > 1 and pares[colunas[1]].nunique() > 1:
            valor = round(float(pares[colunas[0]].corr(pares[colunas[1]], method=metodo)), 3)
        else:
            obs.append("dados insuficientes para calcular a correlação")
        obs.append("correlação mede associação, não causalidade")
        return {**base, "coluna": colunas, "metodo": metodo, "pares_utilizados": int(len(pares)),
                "unidade": "coeficiente de correlação (-1 a 1)", "resultado": valor, "observacoes": obs}

    agregacao = consulta.get("agregacao", "soma") if op in ("ranking", "comparacao") else op
    if agregacao not in AGREGACOES:
        raise ConsultaInvalida(f"agregacao inválida: {agregacao!r}; opções: {sorted(AGREGACOES)}")
    coluna = _validar_coluna(agregacao, consulta.get("coluna"))
    base.update({"agregacao": agregacao, "coluna": coluna, "unidade": _unidade(agregacao, coluna)})
    serie_de = (lambda d: d[coluna] if coluna else d.index.to_series())

    if op == "comparacao":
        comparar = consulta.get("comparar") or {}
        col, valores = comparar.get("coluna"), comparar.get("valores")
        if col not in AGRUPAMENTOS or not isinstance(valores, list) or len(valores) != 2:
            raise ConsultaInvalida(f"comparacao exige comparar={{coluna em {AGRUPAMENTOS}, valores: [a, b]}}")
        grupos = []
        for v in valores:
            m, apl, o = _aplicar_filtros([{"coluna": col, "operador": "eq", "valor": v}])
            parte = DF[mascara & m]
            obs += o
            grupos.append({"valor": v, "valores_no_dataset": apl[0]["valor"], "registros": int(len(parte)),
                           "resultado": _agregar(serie_de(parte), agregacao)})
        a, b = grupos[0]["resultado"], grupos[1]["resultado"]
        dif = pct = None
        if a is not None and b is not None:
            dif = round(b - a, 2)
            if a != 0:
                pct = round(100 * (b - a) / abs(a), 1)
            else:
                obs.append("diferença percentual indefinida: o primeiro grupo tem valor zero")
        if any(g["registros"] == 0 for g in grupos):
            obs.append(_AUSENCIA)
        return {**base, "comparar_por": col, "resultado": {"grupos": grupos, "diferenca_absoluta": dif,
                "diferenca_percentual": pct, "sentido": f"{valores[1]} menos {valores[0]}"}, "observacoes": obs}

    agrupar = consulta.get("agrupar_por") or []
    if isinstance(agrupar, str):
        agrupar = [agrupar]
    if op == "ranking" and not agrupar:
        raise ConsultaInvalida("ranking exige agrupar_por")
    if len(agrupar) > 2 or any(c not in AGRUPAMENTOS for c in agrupar):
        raise ConsultaInvalida(f"agrupar_por aceita até 2 colunas de {AGRUPAMENTOS}")
    base["agrupar_por"] = agrupar
    if sub.empty:
        vazio = [] if agrupar else (0 if agregacao in ("contagem", "contagem_distinta") else None)
        return {**base, "resultado": vazio, "observacoes": obs + [_AUSENCIA]}

    if not agrupar:
        valor = _agregar(serie_de(sub), agregacao)
        res = {**base, "resultado": valor, "observacoes": obs}
        if agregacao in ("minimo", "maximo") and valor is not None:
            res["eventos_correspondentes"] = _extremos(sub, coluna, valor)
        if agregacao == "contagem_distinta" and coluna == "Company":
            sem_caixa = int(sub["Company"].map(_norm).nunique())
            if sem_caixa != valor:
                obs.append(f"ignorando maiúsculas/minúsculas no nome, são {sem_caixa} empresas "
                           "(o dataset grafa algumas empresas de formas diferentes)")
        return res

    ordem = consulta.get("ordem", "desc")
    if ordem not in ("asc", "desc"):
        raise ConsultaInvalida("ordem deve ser 'asc' ou 'desc'")
    limite = consulta.get("limite", 10)
    if not isinstance(limite, int) or isinstance(limite, bool) or not 1 <= limite <= LIMITE_MAXIMO:
        raise ConsultaInvalida(f"limite deve ser inteiro entre 1 e {LIMITE_MAXIMO}")
    chaves = []
    for col in agrupar:
        rotulo, o = _chave_grupo(col, sub[col])
        chaves.append(rotulo.rename(col))
        obs += o
    linhas = []
    for chave, parte in sub.groupby(chaves, sort=False):
        chave = chave if isinstance(chave, tuple) else (chave,)
        linhas.append({**dict(zip(agrupar, chave)), "resultado": _agregar(serie_de(parte), agregacao),
                       "registros": int(len(parte))})
    com_valor = [l for l in linhas if l["resultado"] is not None]
    com_valor.sort(key=lambda l: l["resultado"], reverse=(ordem == "desc"))
    ordenadas = com_valor + [l for l in linhas if l["resultado"] is None]
    return {**base, "ordem": ordem, "limite": limite, "grupos_totais": len(ordenadas),
            "resultado": ordenadas[:limite], "observacoes": obs}


def estatisticas(coluna: str, agrupar_por: str = "") -> dict:
    """Contrato original do kit: resumo (eventos, soma*, média, mediana, mín., máx.) de uma coluna,
    opcionalmente por grupo (até 15 grupos, ordenados pela soma de demitidos ou pela média).
    *soma só para Laid_Off. Implementado sobre consultar_dados()."""
    if coluna not in COLUNAS_CONSULTA:
        return {"erro": f"coluna inválida; opções: {COLUNAS_CONSULTA}"}
    if agrupar_por and agrupar_por not in AGRUPAMENTOS:
        return {"erro": f"agrupamento inválido; opções: {AGRUPAMENTOS}"}
    ags = ["contagem", "media", "mediana", "minimo", "maximo"] + (["soma"] if coluna in SOMAVEIS else [])
    nomes = {"contagem": "eventos", "media": "media", "mediana": "mediana", "minimo": "minimo",
             "maximo": "maximo", "soma": "soma"}
    if not agrupar_por:
        resumo = {nomes[a]: consultar_dados({"operacao": a, "coluna": coluna})["resultado"] for a in ags}
        return {"coluna": coluna, "unidade": UNIDADES[coluna], "resumo": resumo}
    criterio = "soma" if coluna in SOMAVEIS else "media"
    topo = consultar_dados({"operacao": "ranking", "agregacao": criterio, "coluna": coluna,
                            "agrupar_por": [agrupar_por], "limite": 15})
    grupos = []
    for g in topo["resultado"]:
        filtro = [{"coluna": agrupar_por, "operador": "eq", "valor": g[agrupar_por]}] \
            if g[agrupar_por] != "(não informado)" else None
        if filtro is None:
            grupos.append({agrupar_por: g[agrupar_por], criterio: g["resultado"], "eventos": g["registros"]})
            continue
        linha = {agrupar_por: g[agrupar_por]}
        for a in ags:
            linha[nomes[a]] = consultar_dados({"operacao": a, "coluna": coluna, "filtros": filtro})["resultado"]
        grupos.append(linha)
    return {"coluna": coluna, "unidade": UNIDADES[coluna], "agrupado_por": agrupar_por,
            "ordenado_por": criterio, "grupos_totais": topo["grupos_totais"], "por_grupo": grupos,
            "observacoes": topo["observacoes"]}


# ---------------------------------------------------------------- fatos para o RAG
def fatos_do_dataset() -> list[str]:
    """Frases CURTAS com números reais calculados do CSV, uma por tema, no formato 'Tema: texto'.
    São fatos estáticos para o RAG; perguntas com filtros novos devem usar consultar_dados()."""
    y = DF[ALVO].astype(float)
    total = y.sum()
    fatos = [
        f"Visão geral do dataset: {_fmt(len(DF))} eventos de demissão (layoffs), {_fmt(DF['Company'].nunique())} "
        f"empresas distintas (companies) e {_fmt(total)} funcionários demitidos no total (soma de Laid_Off).",
        f"Período do dataset: eventos de {DF['Date_layoffs'].min()} a {DF['Date_layoffs'].max()} (Date_layoffs); "
        f"anos (year) de {int(DF['year'].min())} a {int(DF['year'].max())}.",
        f"Demissões por evento (Laid_Off): média {_fmt(y.mean(), 2)}, mediana {_fmt(y.median())}, mínimo "
        f"{_fmt(y.min())} e máximo {_fmt(y.max())} funcionários demitidos; a média é muito maior que a mediana "
        "por causa de poucos eventos gigantes.",
        f"Valores ausentes (missing values): capital captado (Money_Raised_clean) em "
        f"{_fmt(DF['Money_Raised_clean'].isna().sum())} eventos, estágio (Stage) em {_fmt(DF['Stage'].isna().sum())} "
        f"e estado (USState) em {_fmt(DF['USState'].isna().sum())}; as demais colunas estão completas.",
        f"Divergência temporal: {_fmt(int(_DIVERGE_ANO.sum()))} eventos têm a coluna Year diferente do ano da data "
        "(Date_layoffs); as análises anuais usam year, derivado de Year, como no trabalho anterior.",
    ]
    for ano, g in DF.groupby("year"):
        fatos.append(f"Demissões no ano de {int(ano)} (year {int(ano)}): {_fmt(len(g))} eventos, "
                     f"{_fmt(g['Company'].nunique())} empresas e {_fmt(g[ALVO].sum())} funcionários demitidos "
                     f"({_fmt(100 * g[ALVO].sum() / total, 1)}% do total); média {_fmt(g[ALVO].mean(), 1)} por evento.")
    por_ano = DF.groupby("year")[ALVO].agg(["sum", "count"])
    fatos.append(f"Ano com mais demissões e ranking por ano (year): mais funcionários demitidos em {int(por_ano['sum'].idxmax())} "
                 f"({_fmt(por_ano['sum'].max())}); mais eventos em {int(por_ano['count'].idxmax())} "
                 f"({_fmt(por_ano['count'].max())} eventos); menos eventos em {int(por_ano['count'].idxmin())} "
                 f"({_fmt(por_ano['count'].min())} eventos).")
    for cont, g in DF.groupby("Continent"):
        fatos.append(f"Demissões no continente {cont} (Continent): {_fmt(len(g))} eventos e {_fmt(g[ALVO].sum())} "
                     f"funcionários demitidos ({_fmt(100 * g[ALVO].sum() / total, 1)}% do total).")
    por_pais = DF.groupby("Country")[ALVO].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
    pt = {v[0]: a for a, v in ALIASES["Country"].items() if len(v) == 1 and _norm(a) != _norm(v[0])}
    for i, (pais, r) in enumerate(por_pais.head(10).iterrows(), start=1):
        nome = f"{pais} ({pt[pais]})" if pais in pt else pais
        fatos.append(f"Demissões em {nome} (Country): {_fmt(r['count'])} eventos e {_fmt(r['sum'])} funcionários "
                     f"demitidos, {i}ª posição em total de demitidos ({_fmt(100 * r['sum'] / total, 1)}% do total).")
    fatos.append("País com mais demissões e ranking dos países por funcionários demitidos (Country): " + "; ".join(
        f"{i}º {p} com {_fmt(r['sum'])}" for i, (p, r) in enumerate(por_pais.head(5).iterrows(), start=1)) +
        ". UK e United Kingdom são contados separadamente aqui.")
    fatos.append(f"Países no dataset: {DF['Country'].nunique()} países; {int((DF['Country'].value_counts() < 5).sum())} "
                 "deles têm menos de 5 eventos. O Reino Unido aparece com duas grafias: UK "
                 f"({_fmt((DF['Country'] == 'UK').sum())} eventos) e United Kingdom "
                 f"({_fmt((DF['Country'] == 'United Kingdom').sum())}).")
    por_setor = DF.groupby("Industry")[ALVO].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
    pt_setor = {v[0]: a for a, v in ALIASES["Industry"].items() if len(v) == 1 and _norm(a) != _norm(v[0])}
    for i, (setor, r) in enumerate(por_setor.head(10).iterrows(), start=1):
        nome = f"{setor} ({pt_setor[setor]})" if setor in pt_setor else setor
        fatos.append(f"Demissões no setor {nome} (Industry): {_fmt(r['count'])} eventos e {_fmt(r['sum'])} "
                     f"funcionários demitidos, {i}º setor em total de demitidos; média {_fmt(r['mean'], 1)} por evento.")
    fatos.append("Setor com mais demissões e ranking dos setores por funcionários demitidos (Industry): " + "; ".join(
        f"{i}º {s} com {_fmt(r['sum'])}" for i, (s, r) in enumerate(por_setor.head(5).iterrows(), start=1)) + ".")
    mais_eventos = DF["Industry"].value_counts()
    fatos.append(f"Setor com mais eventos de demissão: {mais_eventos.index[0]} com {_fmt(mais_eventos.iloc[0])} eventos; "
                 f"há {DF['Industry'].nunique()} setores (Industry), {int((mais_eventos < 5).sum())} com menos de 5 eventos.")
    por_estagio = DF.groupby("Stage")[ALVO].agg(["sum", "count"]).sort_values("sum", ascending=False)
    fatos.append("Demissões por estágio de financiamento (Stage), eventos e demitidos: " + "; ".join(
        f"{s} {_fmt(r['count'])} eventos e {_fmt(r['sum'])} demitidos" for s, r in por_estagio.head(6).iterrows()) + ".")
    fatos.append(f"Estágio desconhecido: {_fmt((DF['Stage'] == 'Unknown').sum())} eventos têm Stage = Unknown e "
                 f"{_fmt(DF['Stage'].isna().sum())} não têm estágio registrado.")
    for _, r in DF.nlargest(5, ALVO).iterrows():
        fatos.append(f"Maior evento de demissão (largest layoffs): {r['Company']} demitiu {_fmt(r[ALVO])} funcionários "
                     f"em {r['Date_layoffs']} ({r['Country']}, setor {r['Industry']}).")
    corr = DF[[ALVO, "Company_Size_before_Layoffs", "Money_Raised_clean", "year", "month"]].corr(method="spearman")[ALVO]
    fatos.append("Correlação (Spearman) com o número de demitidos (Laid_Off): porte anterior "
                 f"(Company_Size_before_Layoffs) {_fmt(corr['Company_Size_before_Layoffs'], 3)}, capital captado "
                 f"(Money_Raised_clean) {_fmt(corr['Money_Raised_clean'], 3)}, ano {_fmt(corr['year'], 3)}, "
                 f"mês {_fmt(corr['month'], 3)}. Correlação não implica causalidade.")
    fatos.append(f"Desempenho do modelo no teste: {METRICAS['modelo']}; MAE = {_fmt(METRICAS['mae_teste'], 2)} "
                 f"funcionários; RMSE = {_fmt(METRICAS['rmse_teste'], 2)}; R² = {_fmt(METRICAS['r2_teste'], 4)}; "
                 f"MAE da validação cruzada = {_fmt(METRICAS['mae_cv'], 2)}.")
    return fatos


_conferencia = conferir_transformacao()
if not _conferencia["ok"]:
    log.error("ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino: %s", _conferencia["detalhes"][:3])
