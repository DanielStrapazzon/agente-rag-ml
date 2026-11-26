# Cole esta versão de para_o_modelo() no ferramentas.py, NO LUGAR da original,
# quando as transformações do treino foram feitas FORA do Pipeline.
# (Precisa de `TRANSF = joblib.load(BASE / "modelo" / "transformadores.joblib")` perto do MODELO.)

TRANSF = joblib.load(BASE / "modelo" / "transformadores.joblib")   # scaler + colunas AJUSTADOS no treino


def para_o_modelo(entrada: dict) -> pd.DataFrame:
    """Entrada legível -> formato do modelo, repetindo EXATAMENTE as transformações do treino."""
    X = pd.DataFrame([entrada])[NUMERICAS + CATEGORICAS]
    # Se no treino houve mapa de códigos, ele vem aqui ANTES. Ex.:
    #   X["sexo"] = X["sexo"].map({"masculino": 1, "feminino": 2})
    X = pd.get_dummies(X, columns=CATEGORICAS, dtype=int)          # texto -> 0/1 (igual ao treino)
    X = X.reindex(columns=TRANSF["colunas_modelo"], fill_value=0)   # mesmas colunas, mesma ordem
    X[NUMERICAS] = TRANSF["scaler"].transform(X[NUMERICAS])         # mesma escala (scaler do TREINO)
    return X
