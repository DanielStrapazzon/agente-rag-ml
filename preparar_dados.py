"""
Preparação reproduzível do dataset Tech Layoffs (etapa 02).

Extrai a limpeza da célula 7 de notebooks/analise_modelos.ipynb (a que gerou as métricas do modelo)
e gera um dataset LEGÍVEL: limpo, sem escala, sem encoding, sem imputação e sem log do alvo.
Essas transformações pertencem ao Pipeline de ML e são ajustadas só no treino (etapa 03).

    .venv/bin/python preparar_dados.py
        lê   dados/tech_layoffs.csv                   (original, nunca é alterado)
        gera dados/dataset_legivel.csv                (usado por ferramentas.py)

As funções podem ser importadas pelo futuro script de treino:
    from preparar_dados import carregar_dataset, limpar_dataset, gerar_atributos_derivados,
                               separar_atributos_modelo
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ORIGEM = BASE / "dados" / "tech_layoffs.csv"
DESTINO = BASE / "dados" / "dataset_legivel.csv"          # caminho canônico do kit (etapa 04)

# Colunas exigidas pelo notebook (célula 3). 'Nr' existe no CSV, mas o notebook não a exige.
COLUNAS_OBRIGATORIAS = [
    "Company", "Location_HQ", "Region", "USState", "Country", "Continent", "Laid_Off",
    "Date_layoffs", "Percentage", "Company_Size_before_Layoffs", "Company_Size_after_layoffs",
    "Industry", "Stage", "Money_Raised_in__mil", "Year", "latitude", "longitude",
]
COLUNAS_NUMERICAS_ORIGINAIS = [
    "Nr", "Laid_Off", "Percentage", "Company_Size_before_Layoffs", "Company_Size_after_layoffs",
    "Money_Raised_in__mil", "Year", "latitude", "longitude",
]
DERIVADAS = ["Money_Raised_clean", "year", "month"]

# Seleção de atributos do trabalho anterior (célula 7) — NÃO alterar sem justificar no relatório.
ALVO = "Laid_Off"
CATEGORICAS = ["Country", "Continent", "Industry", "Stage"]
NUMERICAS = ["Company_Size_before_Layoffs", "Money_Raised_clean", "year", "month"]
EXCLUIDAS_VAZAMENTO = ["Percentage", "Company_Size_after_layoffs"]   # derivadas da própria demissão

# Exportadas como inteiro (com ausente preservado) quando todos os valores são inteiros.
COLUNAS_CONTAGEM = ["Nr", "Laid_Off", "Company_Size_before_Layoffs", "Company_Size_after_layoffs",
                    "Year", "year", "month"]
MINIMO_LINHAS = 1000            # requisito acadêmico verificado no notebook


def carregar_dataset(caminho: Path | str = ORIGEM) -> pd.DataFrame:
    """Lê o CSV original como no notebook (low_memory=False, nomes de colunas sem espaços)."""
    df = pd.read_csv(caminho, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def validar_colunas(df: pd.DataFrame, obrigatorias: list[str] = COLUNAS_OBRIGATORIAS) -> None:
    """Levanta ValueError listando as colunas obrigatórias ausentes."""
    ausentes = sorted(set(obrigatorias) - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {ausentes}. Encontradas: {df.columns.tolist()}")


def limpar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza do treino do notebook (célula 7), sem imputar nem remover linhas por campos opcionais:
    - remove linhas duplicadas idênticas;
    - converte o alvo para número e mantém só alvo presente e >= 0;
    - converte as colunas numéricas (texto inválido -> ausente);
    - converte Date_layoffs para data (data inválida -> ausente, a linha é mantida)."""
    limpo = df.drop_duplicates().copy()
    limpo[ALVO] = pd.to_numeric(limpo[ALVO], errors="coerce")
    limpo = limpo.loc[limpo[ALVO].notna() & (limpo[ALVO] >= 0)].copy()
    for col in COLUNAS_NUMERICAS_ORIGINAIS:
        if col in limpo.columns:
            limpo[col] = pd.to_numeric(limpo[col], errors="coerce").astype("float64")
    limpo["Date_layoffs"] = pd.to_datetime(limpo["Date_layoffs"], errors="coerce")
    return limpo


def gerar_atributos_derivados(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos usados pelo modelo anterior, calculados como no treino do notebook:
    Money_Raised_clean = to_numeric(Money_Raised_in__mil); month = mês de Date_layoffs;
    year = coluna Year (e NÃO o ano da data: há 7 linhas em que eles divergem)."""
    saida = df.copy()
    saida["Money_Raised_clean"] = pd.to_numeric(saida["Money_Raised_in__mil"], errors="coerce").astype("float64")
    datas = pd.to_datetime(saida["Date_layoffs"], errors="coerce")
    saida["month"] = datas.dt.month.astype("float64")
    saida["year"] = pd.to_numeric(saida["Year"], errors="coerce").astype("float64")
    return saida


def validar_dataset(df: pd.DataFrame, minimo_linhas: int = MINIMO_LINHAS) -> list[str]:
    """Confere o dataset preparado. Devolve a lista de problemas (vazia = ok)."""
    problemas = []
    faltam = sorted(set(COLUNAS_OBRIGATORIAS + DERIVADAS) - set(df.columns))
    if faltam:
        return [f"colunas ausentes: {faltam}"]
    if len(df) < minimo_linhas:
        problemas.append(f"apenas {len(df)} linhas (mínimo {minimo_linhas})")
    if df.duplicated().any():
        problemas.append(f"{int(df.duplicated().sum())} linhas duplicadas")
    alvo = pd.to_numeric(df[ALVO], errors="coerce")
    if alvo.isna().any() or (alvo < 0).any():
        problemas.append("alvo Laid_Off com valores ausentes ou negativos")
    meses = df["month"].dropna()
    if not meses.between(1, 12).all():
        problemas.append("month fora de 1..12")
    if not any(df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]) for c in CATEGORICAS):
        problemas.append("sem variáveis categóricas em texto")
    return problemas


def separar_atributos_modelo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X e y exatamente como a célula 7 do notebook monta antes do train_test_split:
    categóricas como object com np.nan; numéricas como float64; alvo float64."""
    X = df[CATEGORICAS + NUMERICAS].copy()
    y = pd.to_numeric(df[ALVO], errors="coerce").astype("float64").copy()
    for col in CATEGORICAS:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    for col in NUMERICAS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    return X, y


def para_exportacao(df: pd.DataFrame) -> pd.DataFrame:
    """Formato legível: data AAAA-MM-DD, contagens inteiras (ausente preservado), colunas originais
    na ordem do CSV seguidas das derivadas. Não muda nenhum valor."""
    saida = df.copy()
    saida["Date_layoffs"] = pd.to_datetime(saida["Date_layoffs"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in COLUNAS_CONTAGEM:
        if col in saida.columns:
            valores = saida[col].dropna()
            if (valores % 1 == 0).all():
                saida[col] = saida[col].astype("Int64")
    originais = [c for c in saida.columns if c not in DERIVADAS]
    return saida[originais + DERIVADAS]


def exportar_dataset(df: pd.DataFrame, destino: Path | str = DESTINO) -> Path:
    """Grava o CSV legível em UTF-8, sem índice, com quebra de linha '\\n' (reprodutível)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    para_exportacao(df).to_csv(destino, index=False, encoding="utf-8", lineterminator="\n")
    return destino


def preparar(origem: Path | str = ORIGEM) -> tuple[pd.DataFrame, dict]:
    """Executa carregar -> validar colunas -> limpar -> derivar -> validar. Devolve (df, resumo)."""
    bruto = carregar_dataset(origem)
    validar_colunas(bruto)
    sem_duplicatas = len(bruto.drop_duplicates())
    limpo = gerar_atributos_derivados(limpar_dataset(bruto))
    resumo = {
        "linhas_originais": len(bruto),
        "colunas_originais": bruto.shape[1],
        "duplicatas_removidas": len(bruto) - sem_duplicatas,
        "removidas_por_alvo_invalido": sem_duplicatas - len(limpo),
        "linhas_finais": len(limpo),
        "colunas_finais": limpo.shape[1],
        "datas_invalidas": int(limpo["Date_layoffs"].isna().sum()),
    }
    return limpo, resumo


def main(origem: Path | str = ORIGEM, destino: Path | str = DESTINO, minimo_linhas: int = MINIMO_LINHAS) -> dict:
    df, resumo = preparar(origem)
    problemas = validar_dataset(df, minimo_linhas)
    if problemas:
        raise ValueError(f"Dataset preparado inválido: {problemas}")
    caminho = exportar_dataset(df, destino)
    print(f"Origem : {origem}")
    for chave, valor in resumo.items():
        print(f"  {chave}: {valor}")
    print(f"Gerado : {caminho}")
    return resumo


if __name__ == "__main__":
    main()
