"""Testes do modelo Tech Layoffs (etapa 03).

    .venv/bin/python -m unittest discover -s tests -v

Usam os artefatos JÁ GERADOS por `.venv/bin/python treinar.py` em modelo/ (caminho canônico desde a etapa 04)
(não refazem o GridSearchCV). Implementado com apoio do Claude Code (Anthropic).
"""
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import preparar_dados as P  # noqa: E402
import previsao_layoffs as M  # noqa: E402

PASTA = RAIZ / "modelo"
HASH_ORIGINAL = "a9ef120e81d7206de5be8a56337c0c008792363fec9ad65597610f9d32d81e6a"
# Etapa 04: os artefatos de aluguel foram substituídos pelos do Tech Layoffs (recuperáveis no Git,
# commit 98a8383). Estes hashes são os da etapa 03: se mudarem, o modelo foi retreinado ou alterado.
HASHES_DEFINITIVOS = {
    "modelo/modelo.joblib": "903684f62204ff59a50652859e6330cff472cec7d97ecbd31e6911c8dd549caf",
    "modelo/metricas.json": "048129701c4881f5126e33dc52ae49f262d839ac998d5a0e4af3f134d88afe71",
    "modelo/amostra_conferencia.json": "52b2b182aed3518a45a86c6407b13f939a86fefd25653462de73d8f2ec089fa4",
    "dados/dataset_legivel.csv": "dbe54cfd786fb57f393563b2ad76943f821f882349acccfc270251c413295909",
}
ENTRADAS_NOTEBOOK = ["Country", "Continent", "Industry", "Stage",
                     "Company_Size_before_Layoffs", "Money_Raised_clean", "year", "month"]


def sha256(caminho) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def json_estrito(caminho):
    def recusar(valor):
        raise ValueError(f"constante JSON inválida: {valor}")
    return json.loads(Path(caminho).read_text(encoding="utf-8"), parse_constant=recusar)


@unittest.skipUnless((PASTA / "modelo.joblib").exists(), "rode antes: .venv/bin/python treinar.py")
class TestModeloTechLayoffs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo = M.carregar_modelo()
        cls.metricas = json_estrito(PASTA / "metricas.json")
        cls.amostra = json_estrito(PASTA / "amostra_conferencia.json")
        X, y = P.separar_atributos_modelo(P.preparar()[0])
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te = train_test_split(
            X, y, test_size=cls.metricas["tamanho_teste"], random_state=cls.metricas["random_state"])

    def ttr(self) -> TransformedTargetRegressor:
        return self.modelo.estimador

    # ---------------------------------------------------------------- entradas
    def test_oito_entradas_na_ordem_do_notebook(self):
        self.assertEqual(M.ENTRADAS, ENTRADAS_NOTEBOOK)
        self.assertEqual(self.metricas["entradas_categoricas"] + self.metricas["entradas_numericas"], ENTRADAS_NOTEBOOK)
        self.assertEqual(list(self.ttr().regressor_.named_steps["prep"].feature_names_in_), ENTRADAS_NOTEBOOK)
        self.assertEqual(list(self.X_te.columns), ENTRADAS_NOTEBOOK)

    def test_sem_atributos_de_vazamento(self):
        vistos = set(self.ttr().regressor_.named_steps["prep"].feature_names_in_)
        for col in ["Percentage", "Company_Size_after_layoffs", "Laid_Off"]:
            self.assertNotIn(col, vistos)
            self.assertNotIn(col, M.ENTRADAS)

    def test_tipos_compativeis(self):
        df = M.para_dataframe_modelo(self.amostra[0]["entrada"])
        self.assertEqual(list(df.columns), ENTRADAS_NOTEBOOK)
        for col in P.CATEGORICAS:
            self.assertEqual(df[col].dtype, object)
            self.assertEqual(self.X_te[col].dtype, object)
        for col in P.NUMERICAS:
            self.assertEqual(df[col].dtype, "float64")
            self.assertEqual(self.X_te[col].dtype, "float64")

    def test_entrada_invalida(self):
        with self.assertRaises(ValueError):
            M.para_dataframe_modelo({**self.amostra[0]["entrada"], "month": "abril"})
        with self.assertRaises(ValueError):
            M.para_dataframe_modelo({**self.amostra[0]["entrada"], "Percentage": 10})
        with self.assertRaises(ValueError):
            M.para_dataframe_modelo({**self.amostra[0]["entrada"], "year": float("inf")})

    # ---------------------------------------------------------------- modelo serializado
    def test_carrega_modelo_serializado(self):
        self.assertIsInstance(self.modelo, M.EstimadorNaoNegativo)
        self.assertEqual(type(self.modelo).__module__, "previsao_layoffs", "não pode depender de __main__")
        self.assertFalse(hasattr(self.modelo, "classes_"), "o kit trataria como classificação")

    def test_carrega_em_novo_processo(self):
        import treinar
        self.assertIn("divergencias=0", treinar.validar_em_novo_processo())

    def test_preprocessamento_dentro_do_modelo(self):
        ttr = self.ttr()
        self.assertIsInstance(ttr, TransformedTargetRegressor)
        self.assertIs(ttr.func, np.log1p)
        self.assertIs(ttr.inverse_func, np.expm1)
        prep = ttr.regressor_.named_steps["prep"]
        self.assertIsInstance(prep, ColumnTransformer)
        num = prep.named_transformers_["num"].named_steps
        cat = prep.named_transformers_["cat"].named_steps
        self.assertIsInstance(num["imputer"], SimpleImputer)
        self.assertEqual(num["imputer"].strategy, "median")
        self.assertIsInstance(num["scaler"], StandardScaler)
        self.assertEqual(cat["imputer"].strategy, "most_frequent")
        self.assertIsInstance(cat["encoder"], OneHotEncoder)
        self.assertEqual(cat["encoder"].handle_unknown, "ignore")

    def test_modelo_corresponde_ao_metricas(self):
        nome = type(self.ttr().regressor_.named_steps["model"]).__name__
        self.assertIn({"MLPRegressor": "MLP", "KNeighborsRegressor": "KNN",
                       "LinearRegression": "Regressão Linear"}[nome], self.metricas["modelo"])

    # ---------------------------------------------------------------- previsões
    def test_previsao_valida(self):
        r = M.prever_demissoes(self.amostra[0]["entrada"], self.modelo)
        self.assertIsInstance(r["previsao"], float)
        self.assertTrue(math.isfinite(r["previsao"]) and r["previsao"] >= 0)
        self.assertEqual(r["unidade"], M.UNIDADE_ALVO)
        json.dumps(r, allow_nan=False)                     # tipos nativos

    def test_ausentes_tratados_pelo_pipeline(self):
        entrada = {**self.amostra[0]["entrada"], "Stage": None, "Money_Raised_clean": None, "month": ""}
        r = M.prever_demissoes(entrada, self.modelo)
        self.assertTrue(math.isfinite(r["previsao"]))
        self.assertEqual(sum("não informado" in a for a in r["alertas"]), 3)
        r2 = M.prever_demissoes({k: v for k, v in self.amostra[0]["entrada"].items() if k != "Stage"}, self.modelo)
        self.assertTrue(math.isfinite(r2["previsao"]))

    def test_categoria_desconhecida_gera_alerta(self):
        r = M.prever_demissoes({**self.amostra[0]["entrada"], "Country": "Atlântida"}, self.modelo)
        self.assertTrue(math.isfinite(r["previsao"]))
        self.assertTrue(any("não existe no treino" in a for a in r["alertas"]))

    def test_previsao_acima_do_porte_e_registrada(self):
        r = M.prever_demissoes({**self.amostra[0]["entrada"], "Company_Size_before_Layoffs": 1}, self.modelo)
        self.assertEqual(any("supera o porte" in a for a in r["alertas"]), r["previsao"] > 1)

    def test_previsoes_negativas_viram_zero(self):
        np.testing.assert_array_equal(M.limitar_a_zero([-5.0, -0.1, 0.0, 3.5]), [0.0, 0.0, 0.0, 3.5])
        negativo = M.EstimadorNaoNegativo(DummyRegressor(strategy="constant", constant=-10.0))
        negativo.fit(pd.DataFrame({"a": [1.0, 2.0]}), [1.0, 2.0])
        np.testing.assert_array_equal(negativo.predict(pd.DataFrame({"a": [3.0]})), [0.0])

    def test_sem_nan_ou_infinito_no_teste(self):
        prev = self.modelo.predict(self.X_te)
        self.assertTrue(np.isfinite(prev).all())
        self.assertTrue((prev >= 0).all())

    def test_metricas_reproduzem_o_teste(self):
        from sklearn.metrics import mean_absolute_error, r2_score
        prev = self.modelo.predict(self.X_te)
        self.assertAlmostEqual(mean_absolute_error(self.y_te, prev), self.metricas["mae_teste"], places=2)
        self.assertAlmostEqual(r2_score(self.y_te, prev), self.metricas["r2_teste"], places=4)
        acima = int((prev > self.X_te["Company_Size_before_Layoffs"].to_numpy()).sum())
        self.assertEqual(acima, self.metricas["previsoes_teste_acima_do_porte_anterior"])

    # ---------------------------------------------------------------- amostra e JSON
    def test_amostra_reproduzida(self):
        self.assertGreaterEqual(len(self.amostra), 5)
        for item in self.amostra:
            self.assertEqual(list(item["entrada"]), ENTRADAS_NOTEBOOK)
            obtido = float(self.modelo.predict(M.para_dataframe_modelo(item["entrada"]))[0])
            self.assertAlmostEqual(obtido, item["previsao_no_treino"],
                                   delta=1e-9 * max(1.0, abs(item["previsao_no_treino"])))

    def test_amostra_legivel_e_com_ausente(self):
        for item in self.amostra:
            for col in P.CATEGORICAS:
                self.assertTrue(item["entrada"][col] is None or isinstance(item["entrada"][col], str))
            for col in P.NUMERICAS:
                self.assertTrue(item["entrada"][col] is None or isinstance(item["entrada"][col], (int, float)))
        self.assertTrue(any(None in item["entrada"].values() for item in self.amostra),
                        "a amostra deve cobrir a conversão de null -> np.nan")

    def test_metricas_compativeis_com_kit(self):
        m = self.metricas
        for campo in ["modelo", "tarefa", "r2_teste", "mae_teste", "rmse_teste", "mae_cv", "melhores_parametros",
                      "sklearn_version", "numpy_version", "pandas_version", "joblib_version", "python_version",
                      "random_state", "quantidade_treino", "quantidade_teste", "dataset"]:
            self.assertIn(campo, m)
        self.assertIsInstance(m["modelo"], str)
        self.assertEqual(m["tarefa"], "regressao")
        for campo in ["r2_teste", "mae_teste", "rmse_teste", "mae_cv"]:
            self.assertIsInstance(m[campo], float)
        self.assertEqual(m["sklearn_version"], sklearn.__version__, "ferramentas.py exige a mesma versão")
        self.assertEqual(m["quantidade_treino"] + m["quantidade_teste"], len(self.X_tr) + len(self.X_te))
        self.assertEqual(m["dataset"]["sha256"], HASH_ORIGINAL)

    def test_cv_escolheu_o_modelo(self):
        comp = self.metricas["comparacao_modelos"]
        self.assertEqual(len(comp), 3)
        melhor = min(comp, key=lambda c: c["mae_cv"])
        self.assertTrue(self.metricas["modelo"].startswith(melhor["modelo"]))
        self.assertEqual(self.metricas["mae_cv"], melhor["mae_cv"])

    # ---------------------------------------------------------------- preservação
    def test_dataset_original_preservado(self):
        self.assertEqual(sha256(P.ORIGEM), HASH_ORIGINAL)

    def test_artefatos_definitivos_sao_os_da_etapa_03(self):
        for arquivo, esperado in HASHES_DEFINITIVOS.items():
            self.assertEqual(sha256(RAIZ / arquivo), esperado, arquivo)

    def test_nao_restam_artefatos_de_aluguel(self):
        self.assertNotIn("aluguel", (PASTA / "metricas.json").read_text(encoding="utf-8").lower())
        self.assertNotIn("area_m2", (RAIZ / "dados" / "dataset_legivel.csv").read_text(encoding="utf-8")[:500])


if __name__ == "__main__":
    unittest.main()
