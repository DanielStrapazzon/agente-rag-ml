"""Testes da preparação do dataset Tech Layoffs (etapa 02).

    .venv/bin/python -m unittest discover -s tests -v

Usa DataFrames pequenos e controlados para casos extremos e o CSV real (dados/tech_layoffs.csv)
para validar a compatibilidade com o notebook. Não grava nada em dados/: as exportações vão para
um diretório temporário.
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import preparar_dados as P  # noqa: E402

HASH_ORIGINAL = "a9ef120e81d7206de5be8a56337c0c008792363fec9ad65597610f9d32d81e6a"


def sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def linha(**alteracoes) -> dict:
    """Uma linha válida (valores da 1ª linha do CSV real), com alterações pontuais."""
    base = {"Nr": 1, "Company": "Tamara Mellon", "Location_HQ": "Los Angeles", "Region": "other",
            "USState": "California", "Country": "USA", "Continent": "North America", "Laid_Off": "20.0",
            "Date_layoffs": "2020-03-12", "Percentage": "40.0", "Company_Size_before_Layoffs": "50.0",
            "Company_Size_after_layoffs": "30.0", "Industry": "Retail", "Stage": "Series C",
            "Money_Raised_in__mil": "90.0", "Year": "2020", "latitude": "34.05", "longitude": "-118.24"}
    base.update(alteracoes)
    return base


def df_controlado(*linhas) -> pd.DataFrame:
    return pd.DataFrame(list(linhas)).astype(str).replace({"None": np.nan})


def reproducao_celula7_notebook(df: pd.DataFrame):
    """Cópia literal da preparação da célula 7 de notebooks/analise_modelos.ipynb (referência)."""
    work = df.drop_duplicates().copy()
    work['Laid_Off'] = pd.to_numeric(work['Laid_Off'], errors='coerce')
    work = work.loc[work['Laid_Off'].notna() & (work['Laid_Off'] >= 0)].copy()
    work['Money_Raised_clean'] = pd.to_numeric(work['Money_Raised_in__mil'], errors='coerce')
    work['Company_Size_before_Layoffs'] = pd.to_numeric(work['Company_Size_before_Layoffs'], errors='coerce')
    work['Date_layoffs'] = pd.to_datetime(work['Date_layoffs'], errors='coerce')
    work['month'] = work['Date_layoffs'].dt.month
    work['year'] = pd.to_numeric(work['Year'], errors='coerce')
    cats = ['Country', 'Continent', 'Industry', 'Stage']
    nums = ['Company_Size_before_Layoffs', 'Money_Raised_clean', 'year', 'month']
    X = work[cats + nums].copy()
    y = work['Laid_Off'].astype('float64').copy()
    for col in cats:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    for col in nums:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64')
    return X, y


class TestDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hash_antes = sha256(P.ORIGEM)
        cls.bruto = P.carregar_dataset()
        cls.preparado, cls.resumo = P.preparar()

    # 1. carregamento
    def test_01_carrega_dataset_original(self):
        self.assertEqual(self.bruto.shape, (1745, 18))
        P.validar_colunas(self.bruto)                      # não levanta
        self.assertEqual(self.resumo["linhas_finais"], 1745)
        self.assertEqual(P.validar_dataset(self.preparado), [])

    # 3/4. datas e atributos temporais no dado real
    def test_03_datas_reais_validas(self):
        datas = self.preparado["Date_layoffs"]
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(datas))
        self.assertEqual(int(datas.isna().sum()), 0)
        self.assertEqual(str(datas.min().date()), "2020-03-12")
        self.assertEqual(str(datas.max().date()), "2025-12-28")
        self.assertTrue((self.preparado["month"] == datas.dt.month).all())
        self.assertTrue((self.preparado["year"] == self.bruto["Year"]).all())

    # 6. categorias preservadas
    def test_06_categorias_reais_preservadas(self):
        for col in ["Company", "Country", "Continent", "Industry", "Stage", "USState", "Region", "Location_HQ"]:
            pd.testing.assert_series_equal(self.preparado[col].reset_index(drop=True),
                                           self.bruto[col].reset_index(drop=True), check_dtype=False)

    # 9. ausentes preservados
    def test_09_ausentes_reais_preservados(self):
        self.assertEqual(int(self.preparado["Money_Raised_in__mil"].isna().sum()), 110)
        self.assertEqual(int(self.preparado["Money_Raised_clean"].isna().sum()), 110)
        self.assertEqual(int(self.preparado["Stage"].isna().sum()), 27)
        self.assertEqual(int(self.preparado["USState"].isna().sum()), 1)

    # 10. compatibilidade com o notebook
    def test_10_compatibilidade_com_notebook(self):
        X_nb, y_nb = reproducao_celula7_notebook(self.bruto)
        X, y = P.separar_atributos_modelo(self.preparado)
        self.assertEqual(list(X.columns), P.CATEGORICAS + P.NUMERICAS)
        pd.testing.assert_frame_equal(X, X_nb)
        pd.testing.assert_series_equal(y, y_nb)
        for col in P.EXCLUIDAS_VAZAMENTO:
            self.assertNotIn(col, X.columns)

    def test_10b_csv_legivel_reproduz_entradas_do_notebook(self):
        """A etapa 03 poderá treinar a partir do CSV legível: X/y relidos têm que ser os mesmos."""
        X_nb, y_nb = reproducao_celula7_notebook(self.bruto)
        with tempfile.TemporaryDirectory() as tmp:
            destino = P.exportar_dataset(self.preparado, Path(tmp) / "legivel.csv")
            relido = pd.read_csv(destino, low_memory=False)
        X, y = P.separar_atributos_modelo(relido)
        pd.testing.assert_frame_equal(X, X_nb)
        pd.testing.assert_series_equal(y, y_nb, check_names=True)

    def test_10c_exportacao_nao_altera_valores_originais(self):
        with tempfile.TemporaryDirectory() as tmp:
            relido = pd.read_csv(P.exportar_dataset(self.preparado, Path(tmp) / "l.csv"), low_memory=False)
        original = self.bruto
        self.assertEqual(list(relido.columns), list(original.columns) + P.DERIVADAS)
        for col in original.columns:
            if pd.api.types.is_numeric_dtype(original[col]):
                np.testing.assert_allclose(relido[col].astype(float), original[col].astype(float),
                                           rtol=0, atol=0, equal_nan=True, err_msg=col)
            else:
                pd.testing.assert_series_equal(relido[col], original[col], check_dtype=False)

    # 11. reprodutibilidade
    def test_11_exportacao_reprodutivel(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = P.main(destino=Path(tmp) / "a.csv")
            b = P.main(destino=Path(tmp) / "b.csv")
            self.assertEqual(a, b)
            self.assertEqual(sha256(Path(tmp) / "a.csv"), sha256(Path(tmp) / "b.csv"))
            texto = (Path(tmp) / "a.csv").read_bytes()
        self.assertNotIn(b"\r\n", texto)
        texto.decode("utf-8")                                 # UTF-8 válido

    # 12. original preservado
    def test_12_csv_original_preservado(self):
        with tempfile.TemporaryDirectory() as tmp:
            P.main(destino=Path(tmp) / "x.csv")
        self.assertEqual(sha256(P.ORIGEM), self.hash_antes)
        self.assertEqual(sha256(P.ORIGEM), HASH_ORIGINAL)


class TestCasosControlados(unittest.TestCase):
    # 2. colunas ausentes
    def test_02_coluna_obrigatoria_ausente(self):
        df = df_controlado(linha()).drop(columns=["Industry", "Stage"])
        with self.assertRaises(ValueError) as ctx:
            P.validar_colunas(df)
        self.assertIn("Industry", str(ctx.exception))
        self.assertIn("Stage", str(ctx.exception))

    def test_02b_carregar_remove_espacos_dos_nomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "c.csv"
            df_controlado(linha()).rename(columns={"Industry": " Industry "}).to_csv(arq, index=False)
            self.assertIn("Industry", P.carregar_dataset(arq).columns)

    # 3. datas
    def test_03_conversao_de_datas(self):
        df = P.limpar_dataset(df_controlado(linha(), linha(Nr=2, Date_layoffs="data-invalida"),
                                            linha(Nr=3, Date_layoffs=None)))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date_layoffs"]))
        self.assertEqual(df["Date_layoffs"].iloc[0], pd.Timestamp("2020-03-12"))
        self.assertEqual(len(df), 3, "data inválida não pode remover a linha")
        self.assertEqual(int(df["Date_layoffs"].isna().sum()), 2)
        exportado = P.para_exportacao(P.gerar_atributos_derivados(df))
        self.assertEqual(exportado["Date_layoffs"].iloc[0], "2020-03-12")
        self.assertTrue(pd.isna(exportado["Date_layoffs"].iloc[1]))

    # 4. year e month
    def test_04_year_e_month(self):
        df = P.gerar_atributos_derivados(P.limpar_dataset(df_controlado(
            linha(Date_layoffs="2022-11-30", Year="2022"),
            linha(Nr=2, Date_layoffs="2020-04-01", Year="2024"),      # divergência real do CSV (Acko)
            linha(Nr=3, Date_layoffs="xx", Year="2023"))))
        self.assertEqual(df["month"].tolist()[:2], [11.0, 4.0])
        self.assertTrue(np.isnan(df["month"].iloc[2]))
        self.assertEqual(df["year"].tolist(), [2022.0, 2024.0, 2023.0], "year vem de Year, como no notebook")

    # 5. capital captado
    def test_05_capital_captado(self):
        df = P.gerar_atributos_derivados(P.limpar_dataset(df_controlado(
            linha(Money_Raised_in__mil="121900.0"), linha(Nr=2, Money_Raised_in__mil=None),
            linha(Nr=3, Money_Raised_in__mil="$1,200"), linha(Nr=4, Money_Raised_in__mil="1e3"))))
        self.assertEqual(df["Money_Raised_clean"].dtype, "float64")
        self.assertEqual(df["Money_Raised_clean"].iloc[0], 121900.0)
        self.assertTrue(np.isnan(df["Money_Raised_clean"].iloc[1]))
        self.assertTrue(np.isnan(df["Money_Raised_clean"].iloc[2]), "como no treino: texto vira ausente")
        self.assertEqual(df["Money_Raised_clean"].iloc[3], 1000.0)
        self.assertNotEqual(df["Money_Raised_clean"].iloc[2], 0, "ausente nunca vira zero")

    # 6. categorias
    def test_06_categorias_preservadas(self):
        df = P.limpar_dataset(df_controlado(linha(Country="United Kingdom", Industry="Transportion"),
                                            linha(Nr=2, Country="UK", Stage=None, Company="FreshBooks")))
        self.assertEqual(df["Country"].tolist(), ["United Kingdom", "UK"], "sem normalização de grafia")
        self.assertEqual(df["Industry"].iloc[0], "Transportion")
        self.assertEqual(df["Company"].iloc[1], "FreshBooks")
        X, _ = P.separar_atributos_modelo(P.gerar_atributos_derivados(df))
        self.assertEqual(X["Stage"].dtype, object)
        self.assertTrue(X["Stage"].iloc[1] is np.nan)

    # 7. duplicatas
    def test_07_duplicatas(self):
        df = P.limpar_dataset(df_controlado(linha(), linha(), linha(Nr=2)))
        self.assertEqual(len(df), 2, "só linhas idênticas são removidas")
        _, resumo = self._preparar_controlado(linha(), linha(), linha(Nr=2))
        self.assertEqual(resumo["duplicatas_removidas"], 1)

    # 8. alvo
    def test_08_variavel_alvo(self):
        df = P.limpar_dataset(df_controlado(linha(), linha(Nr=2, Laid_Off="-5"), linha(Nr=3, Laid_Off=None),
                                            linha(Nr=4, Laid_Off="abc"), linha(Nr=5, Laid_Off="0")))
        self.assertEqual(df["Nr"].tolist(), [1.0, 5.0])
        self.assertEqual(df["Laid_Off"].dtype, "float64")
        problemas = P.validar_dataset(P.gerar_atributos_derivados(
            df.assign(Laid_Off=[20.0, -1.0])), minimo_linhas=1)
        self.assertTrue(any("alvo" in p for p in problemas))

    # 9. ausentes
    def test_09_ausentes_nao_removem_linhas_nem_viram_zero(self):
        df = P.gerar_atributos_derivados(P.limpar_dataset(df_controlado(
            linha(), linha(Nr=2, Stage=None, Money_Raised_in__mil=None, USState=None,
                           Company_Size_before_Layoffs=None))))
        self.assertEqual(len(df), 2)
        X, _ = P.separar_atributos_modelo(df)
        self.assertEqual(int(X.iloc[1].isna().sum()), 3)
        self.assertNotIn(0.0, X.iloc[1][["Company_Size_before_Layoffs", "Money_Raised_clean"]].tolist())
        exportado = P.para_exportacao(df)
        self.assertTrue(pd.isna(exportado.loc[exportado["Nr"] == 2, "Company_Size_before_Layoffs"]).all())

    def test_09b_validacao_minimo_de_linhas(self):
        df = P.gerar_atributos_derivados(P.limpar_dataset(df_controlado(linha())))
        self.assertTrue(any("mínimo" in p for p in P.validar_dataset(df)))
        self.assertEqual(P.validar_dataset(df, minimo_linhas=1), [])

    def _preparar_controlado(self, *linhas):
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "c.csv"
            df_controlado(*linhas).to_csv(arq, index=False)
            return P.preparar(arq)


class TestSemEfeitosColaterais(unittest.TestCase):
    def test_importar_nao_executa_processamento(self):
        import importlib
        destino = P.DESTINO
        mtime = destino.stat().st_mtime_ns if destino.exists() else None
        importlib.reload(P)
        self.assertEqual(destino.stat().st_mtime_ns if destino.exists() else None, mtime)


if __name__ == "__main__":
    unittest.main()
