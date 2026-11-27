"""Testes de ferramentas.py (etapa 04): dataset e modelo definitivos, validação, aliases, previsão,
casos semelhantes e consultas estruturadas.

    .venv/bin/python -m unittest discover -s tests -v

As consultas são comparadas com cálculos INDEPENDENTES feitos com pandas diretamente sobre o CSV
ORIGINAL (dados/tech_layoffs.csv), nunca com o resultado da própria função.
Implementado com apoio do Claude Code (Anthropic).
"""
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("GEMINI_API_KEY", "teste-offline")       # agente.py exige a variável; o Gemini não é chamado

import ferramentas as F  # noqa: E402
import previsao_layoffs as M  # noqa: E402

BRUTO = pd.read_csv(RAIZ / "dados" / "tech_layoffs.csv", low_memory=False)     # referência independente
BASE = {"Country": "USA", "Continent": "North America", "Industry": "Retail", "Stage": "Series C",
        "Company_Size_before_Layoffs": "500", "Money_Raised_clean": "90", "year": "2023", "month": "4"}


def json_estrito(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


def consultar(q):
    r = F.consultar_dados(q)
    json_estrito(r)
    return r


class TestArtefatosDefinitivos(unittest.TestCase):
    def test_dataset_definitivo(self):
        self.assertEqual(F.DF.shape, (1745, 21))
        self.assertEqual(int(F.DF["Laid_Off"].sum()), int(BRUTO["Laid_Off"].sum()))
        self.assertIn("Money_Raised_clean", F.DF.columns)
        self.assertNotIn("aluguel", F.DF.columns)

    def test_modelo_definitivo(self):
        self.assertIsInstance(F.MODELO, M.EstimadorNaoNegativo)
        self.assertTrue(F.METRICAS["modelo"].startswith("MLP"))
        self.assertEqual((F.METRICAS["r2_teste"], F.METRICAS["mae_teste"], F.METRICAS["rmse_teste"],
                          F.METRICAS["mae_cv"]), (0.5537, 236.03, 851.53, 201.19))   # iguais à etapa 03

    def test_ponte_reproduz_o_treino(self):
        self.assertEqual(F.conferir_transformacao(), {"ok": True, "detalhes": []})
        amostra = json.loads((RAIZ / "modelo" / "amostra_conferencia.json").read_text(encoding="utf-8"))
        self.assertTrue(any(None in a["entrada"].values() for a in amostra), "a amostra cobre ausentes")

    def test_ponte_nao_transforma_fora_do_pipeline(self):
        df = F.para_o_modelo({**{k: None for k in F.ATRIBUTOS}, "Country": "Brazil", "year": 2023})
        self.assertEqual(list(df.columns), M.ENTRADAS)
        self.assertEqual(df.at[0, "year"], 2023.0)                  # escala original, sem StandardScaler
        self.assertTrue(np.isnan(df.at[0, "month"]))                # ausente -> NaN (imputado no Pipeline)
        self.assertEqual(df.at[0, "Country"], "Brazil")             # texto, sem one-hot


class TestContratoComAgente(unittest.TestCase):
    def test_contratos_usados_pelo_agente(self):
        import agente
        esquema = agente._esquema_interpretacao()
        json.dumps(esquema)
        self.assertEqual(list(F.ATRIBUTOS), M.ENTRADAS)
        for c in F.CATEGORICAS:
            self.assertTrue(F.ATRIBUTOS[c]["valores"])
        self.assertEqual({c: list(F.FAIXAS[c]) for c in F.NUMERICAS}["year"], [2020.0, 2025.0])
        self.assertTrue(all(isinstance(c, str) for c in F.COLUNAS_CONSULTA + F.AGRUPAMENTOS))
        self.assertIn("Laid_Off", esquema["properties"]["consulta"]["properties"]["coluna"]["enum"])
        r = F.estatisticas("Laid_Off", "Country")
        json_estrito(r)
        self.assertEqual(r["por_grupo"][0]["Country"], "USA")


class TestValidacao(unittest.TestCase):
    def test_entrada_valida(self):
        e, erros, alertas = F.preparar(BASE)
        self.assertEqual(erros, [])
        self.assertEqual(e["Company_Size_before_Layoffs"], 500)
        self.assertEqual(e["month"], 4)

    def test_entrada_incompleta(self):
        self.assertEqual(F.faltando({**BASE, "year": None, "Stage": ""}), ["Stage", "year"])
        self.assertEqual(F.faltando({**BASE, "Continent": None}), [], "continente é deduzido do país")
        self.assertIn("Continent", F.faltando({**BASE, "Country": None, "Continent": None}))
        self.assertEqual(F.faltando({**BASE, "Stage": "não sei"}), [], "desconhecido é uma resposta")

    def test_continente_deduzido(self):
        e, erros, alertas = F.preparar({**BASE, "Country": "Brasil", "Continent": None})
        self.assertEqual((e["Country"], e["Continent"]), ("Brazil", "South America"))
        self.assertTrue(any("deduzido" in a for a in alertas))

    def test_entradas_invalidas(self):
        for campo, valor in [("Company_Size_before_Layoffs", "12 abc"), ("Company_Size_before_Layoffs", "doze"),
                             ("year", "dois mil"), ("month", "13º"), ("Money_Raised_clean", "500 mil"),
                             ("Company_Size_before_Layoffs", 10.5), ("Company_Size_before_Layoffs", True),
                             ("year", float("inf")), ("month", float("nan"))]:
            with self.subTest(campo=campo, valor=valor):
                self.assertTrue(F.preparar({**BASE, campo: valor})[1])

    def test_valores_impossiveis(self):
        casos = {"month": "13", "Company_Size_before_Layoffs": "0", "Money_Raised_clean": "-1", "year": "1800"}
        for campo, valor in casos.items():
            with self.subTest(campo=campo):
                erros = F.preparar({**BASE, campo: valor})[1]
                self.assertEqual(len(erros), 1, erros)
        self.assertTrue(F.preparar({**BASE, "Company_Size_before_Layoffs": "-5"})[1])

    def test_conversao_numerica(self):
        self.assertEqual(F.converter_numero("Company_Size_before_Layoffs", "5.000")[0], 5000)
        self.assertEqual(F.converter_numero("Company_Size_before_Layoffs", "5,5 mil")[0], 5500)
        self.assertEqual(F.converter_numero("Company_Size_before_Layoffs", "1200 funcionários")[0], 1200)
        self.assertEqual(F.converter_numero("Money_Raised_clean", "1,2 bilhão")[0], 1200)
        self.assertEqual(F.converter_numero("Money_Raised_clean", "300 milhões")[0], 300)
        self.assertTrue(F.converter_numero("Money_Raised_clean", "US$ 300")[2], "moeda gera alerta")
        self.assertEqual(F.converter_numero("month", "Abril")[0], 4)
        self.assertEqual(F.converter_numero("month", "sep")[0], 9)

    def test_fora_da_faixa_gera_alerta(self):
        e, erros, alertas = F.preparar({**BASE, "Company_Size_before_Layoffs": "900000", "year": "2027"})
        self.assertEqual(erros, [])
        self.assertEqual(sum("fora da faixa" in a for a in alertas), 2)

    def test_opcional_desconhecido(self):
        e, erros, alertas = F.preparar({**BASE, "Stage": "não sei", "Money_Raised_clean": "desconhecido"})
        self.assertEqual(erros, [])
        self.assertIsNone(e["Stage"])
        self.assertIsNone(e["Money_Raised_clean"])
        self.assertTrue(any("Post-IPO" in a for a in alertas))
        self.assertTrue(math.isfinite(F.prever(e)["previsao"]))
        self.assertEqual(F.preparar({**BASE, "Stage": "Unknown"})[0]["Stage"], "Unknown", "categoria real do dataset")

    def test_obrigatorio_desconhecido_e_erro(self):
        erros = F.preparar({**BASE, "Company_Size_before_Layoffs": "não sei"})[1]
        self.assertTrue(any("obrigatório" in e for e in erros))

    def test_categoria_inexistente(self):
        erros = F.preparar({**BASE, "Industry": "Mineração lunar"})[1]
        self.assertTrue(any("não existe" in e for e in erros))

    def test_categoria_fora_do_treino_gera_alerta(self):
        e, erros, alertas = F.preparar({**BASE, "Country": "South Korea", "Continent": None})
        self.assertEqual(erros, [])
        self.assertTrue(any("não apareceu no treino" in a for a in alertas))

    def test_pais_continente_incompativeis(self):
        erros = F.preparar({**BASE, "Country": "Brasil", "Continent": "Europa"})[1]
        self.assertTrue(any("incompatíveis" in e for e in erros))


class TestAliases(unittest.TestCase):
    def test_aliases_previsao(self):
        esperado = {"Brasil": "Brazil", "Estados Unidos": "USA", "EUA": "USA", "Alemanha": "Germany",
                    "índia": "India", "BRAZIL": "Brazil", "alemanha": "Germany"}
        for alias, valor in esperado.items():
            with self.subTest(alias=alias):
                self.assertEqual(F.resolver_categoria("Country", alias, "previsao")["valores"], [valor])
        self.assertEqual(F.resolver_categoria("Continent", "América do Norte", "previsao")["valores"], ["North America"])
        self.assertEqual(F.resolver_categoria("Continent", "america do sul", "previsao")["valores"], ["South America"])
        self.assertEqual(F.resolver_categoria("Stage", "pós-IPO", "previsao")["valores"], ["Post-IPO"])
        self.assertEqual(F.resolver_categoria("Industry", "varejo", "previsao")["valores"], ["Retail"])

    def test_alias_ambiguo_na_previsao(self):
        r = F.resolver_categoria("Country", "Reino Unido", "previsao")
        self.assertIsNotNone(r["erro"])
        self.assertEqual(r["valores"], ["UK", "United Kingdom"])
        self.assertEqual(F.resolver_categoria("Country", "UK", "previsao")["valores"], ["UK"], "UK exato não é ambíguo")

    def test_alias_agrupa_na_consulta(self):
        r = F.resolver_categoria("Country", "Reino Unido", "consulta")
        self.assertEqual(r["valores"], ["UK", "United Kingdom"])
        self.assertIsNotNone(r["observacao"])
        self.assertEqual(F.resolver_categoria("Country", "UK", "consulta")["valores"], ["UK", "United Kingdom"])
        self.assertEqual(F.resolver_categoria("Industry", "Transportation", "consulta")["valores"],
                         ["Transportation", "Transportion"])


class TestPrevisao(unittest.TestCase):
    def test_previsao_valida(self):
        e = F.preparar(BASE)[0]
        p = F.prever(e)
        json_estrito(p)
        self.assertAlmostEqual(p["previsao"], round(float(F.MODELO.predict(F.para_o_modelo(e))[0]), 2))
        self.assertEqual(p["erro_absoluto_medio_no_teste_MAE"], 236.03)
        self.assertIn("NÃO é intervalo de confiança", p["aviso_sobre_a_faixa"])
        self.assertIn("não é a probabilidade", p["o_que_e"])
        self.assertFalse(p["previsao_inconsistente"])

    def test_previsao_acima_do_porte(self):
        e = F.preparar({**BASE, "Company_Size_before_Layoffs": "6"})[0]
        p = F.prever(e)
        bruto = float(F.MODELO.predict(F.para_o_modelo(e))[0])
        self.assertGreater(bruto, 6)
        self.assertTrue(p["previsao_inconsistente"])
        self.assertAlmostEqual(p["previsao"], round(bruto, 2), msg="o valor não é limitado ao porte")
        self.assertIn("não é uma quantidade plausível", p["aviso_inconsistencia"])


class TestCasosSemelhantes(unittest.TestCase):
    def test_casos_semelhantes(self):
        e = F.preparar(BASE)[0]
        prev = F.prever(e)["previsao"]
        c = F.casos_semelhantes(e, prev)
        json_estrito(c)
        self.assertEqual(c["quantidade_casos"], 5)
        reais = [caso["Laid_Off"] for caso in c["casos"]]
        self.assertAlmostEqual(c["media_real_dos_casos"], round(float(np.mean(reais)), 2))
        self.assertAlmostEqual(c["mediana_real_dos_casos"], round(float(np.median(reais)), 2))
        self.assertEqual(c["faixa_real_dos_casos"], [min(reais), max(reais)])
        self.assertEqual(c["previsao_dentro_da_faixa"], min(reais) <= prev <= max(reais))
        dist = [caso["distancia"] for caso in c["casos"]]
        self.assertEqual(dist, sorted(dist))
        self.assertIn("não uma validação independente", c["observacao"])

    def test_casos_semelhantes_com_ausentes(self):
        e = F.preparar({**BASE, "Stage": "não sei", "Money_Raised_clean": "desconhecido"})[0]
        c = F.casos_semelhantes(e, 100.0)
        json_estrito(c)
        self.assertEqual(c["quantidade_casos"], 5)
        self.assertTrue(all(0 <= caso["distancia"] <= 1 for caso in c["casos"]))
        d = F.distancia_gower(e)
        self.assertFalse(d.isna().any(), "eventos com Stage/capital ausentes também recebem distância")

    def test_gower_identico_e_zero(self):
        linha = F.DF.iloc[0]
        e = {a: F._nativo(linha[a]) for a in F.ATRIBUTOS}
        self.assertEqual(float(F.distancia_gower(e).iloc[0]), 0.0)

    def test_fallback_informado(self):
        e = F.preparar({**BASE, "Country": "Senegal", "Continent": None, "Industry": "Aerospace",
                        "Stage": "Series J", "Company_Size_before_Layoffs": "600000"})[0]
        c = F.casos_semelhantes(e)
        if c["fallback"]:
            self.assertIn("aviso_fallback", c)


class TestConsultas(unittest.TestCase):
    def test_soma_de_demissoes(self):
        self.assertEqual(consultar({"operacao": "soma", "coluna": "Laid_Off"})["resultado"], float(BRUTO["Laid_Off"].sum()))

    def test_contagem_de_eventos(self):
        r = consultar({"operacao": "contagem"})
        self.assertEqual(r["resultado"], len(BRUTO))
        self.assertEqual(r["unidade"], "eventos de demissão")

    def test_contagem_distinta_de_empresas(self):
        r = consultar({"operacao": "contagem_distinta", "coluna": "Company"})
        self.assertEqual(r["resultado"], BRUTO["Company"].nunique())
        self.assertTrue(any(str(BRUTO["Company"].str.lower().nunique()) in o for o in r["observacoes"]))

    def test_filtro_pais(self):
        r = consultar({"operacao": "soma", "coluna": "Laid_Off",
                       "filtros": [{"coluna": "Country", "operador": "eq", "valor": "Brasil"}]})
        self.assertEqual(r["resultado"], float(BRUTO.loc[BRUTO["Country"] == "Brazil", "Laid_Off"].sum()))
        self.assertEqual(r["registros_encontrados"], int((BRUTO["Country"] == "Brazil").sum()))

    def test_filtro_empresa(self):
        r = consultar({"operacao": "soma", "coluna": "Laid_Off",
                       "filtros": [{"coluna": "Company", "operador": "eq", "valor": "intel"}]})
        self.assertEqual(r["resultado"], float(BRUTO.loc[BRUTO["Company"] == "Intel", "Laid_Off"].sum()))
        r = consultar({"operacao": "contagem", "filtros": [{"coluna": "Company", "operador": "eq", "valor": "clearco"}]})
        self.assertEqual(r["resultado"], int(BRUTO["Company"].str.lower().eq("clearco").sum()))

    def test_filtro_ano(self):
        r = consultar({"operacao": "soma", "coluna": "Laid_Off",
                       "filtros": [{"coluna": "year", "operador": "eq", "valor": 2023}]})
        self.assertEqual(r["resultado"], float(BRUTO.loc[BRUTO["Year"] == 2023, "Laid_Off"].sum()))

    def test_filtros_combinados(self):
        r = consultar({"operacao": "media", "coluna": "Laid_Off", "filtros": [
            {"coluna": "Country", "operador": "eq", "valor": "Estados Unidos"},
            {"coluna": "year", "operador": "between", "valor": [2022, 2023]},
            {"coluna": "Industry", "operador": "in", "valor": ["Retail", "Finance"]},
            {"coluna": "Company_Size_before_Layoffs", "operador": "gte", "valor": 1000}]})
        m = (BRUTO["Country"].eq("USA") & BRUTO["Year"].between(2022, 2023) &
             BRUTO["Industry"].isin(["Retail", "Finance"]) & BRUTO["Company_Size_before_Layoffs"].ge(1000))
        self.assertEqual(r["registros_encontrados"], int(m.sum()))
        self.assertAlmostEqual(r["resultado"], round(float(BRUTO.loc[m, "Laid_Off"].mean()), 2))

    def test_filtro_por_data(self):
        r = consultar({"operacao": "contagem", "filtros": [
            {"coluna": "Date_layoffs", "operador": "between", "valor": ["2023-01-01", "2023-01-31"]}]})
        datas = pd.to_datetime(BRUTO["Date_layoffs"])
        self.assertEqual(r["resultado"], int(datas.between("2023-01-01", "2023-01-31").sum()))
        self.assertTrue(any("Date_layoffs" in o for o in r["observacoes"]))

    def test_agrupamento(self):
        r = consultar({"operacao": "soma", "coluna": "Laid_Off", "agrupar_por": ["year"], "ordem": "asc", "limite": 10})
        esperado = BRUTO.groupby("Year")["Laid_Off"].sum().sort_values()
        self.assertEqual([(g["year"], g["resultado"]) for g in r["resultado"]],
                         [(int(k), float(v)) for k, v in esperado.items()])
        self.assertTrue(any("Year" in o for o in r["observacoes"]), "avisa as 7 divergências Year x data")

    def test_ranking(self):
        r = consultar({"operacao": "ranking", "agregacao": "soma", "coluna": "Laid_Off",
                       "agrupar_por": ["Industry"], "limite": 3})
        esperado = BRUTO.groupby("Industry")["Laid_Off"].sum().nlargest(3)
        self.assertEqual([g["Industry"] for g in r["resultado"]], list(esperado.index))
        self.assertEqual([g["resultado"] for g in r["resultado"]], [float(v) for v in esperado])

    def test_ranking_agrupa_grafias(self):
        r = consultar({"operacao": "ranking", "agregacao": "soma", "coluna": "Laid_Off",
                       "agrupar_por": ["Country"], "limite": 50})
        uk = next(g for g in r["resultado"] if g["Country"] == "UK")
        self.assertEqual(uk["resultado"], float(BRUTO.loc[BRUTO["Country"].isin(["UK", "United Kingdom"]), "Laid_Off"].sum()))
        self.assertNotIn("United Kingdom", [g["Country"] for g in r["resultado"]])
        self.assertTrue(any("agrupadas" in o for o in r["observacoes"]))

    def test_comparacao(self):
        r = consultar({"operacao": "comparacao", "agregacao": "soma", "coluna": "Laid_Off",
                       "comparar": {"coluna": "year", "valores": [2022, 2023]}})
        a = float(BRUTO.loc[BRUTO["Year"] == 2022, "Laid_Off"].sum())
        b = float(BRUTO.loc[BRUTO["Year"] == 2023, "Laid_Off"].sum())
        self.assertEqual(r["resultado"]["diferenca_absoluta"], round(b - a, 2))
        self.assertEqual(r["resultado"]["diferenca_percentual"], round(100 * (b - a) / a, 1))
        self.assertEqual([g["registros"] for g in r["resultado"]["grupos"]],
                         [int((BRUTO["Year"] == 2022).sum()), int((BRUTO["Year"] == 2023).sum())])

    def test_comparacao_com_zero(self):
        r = consultar({"operacao": "comparacao", "agregacao": "contagem",
                       "comparar": {"coluna": "year", "valores": [2021, 2022]},
                       "filtros": [{"coluna": "Country", "operador": "eq", "valor": "Brazil"}]})
        if r["resultado"]["grupos"][0]["resultado"] == 0:
            self.assertIsNone(r["resultado"]["diferenca_percentual"])
            self.assertTrue(any("indefinida" in o for o in r["observacoes"]))

    def test_correlacao(self):
        r = consultar({"operacao": "correlacao", "colunas": ["Laid_Off", "Company_Size_before_Layoffs"],
                       "metodo": "spearman"})
        self.assertAlmostEqual(r["resultado"], round(float(BRUTO["Laid_Off"].corr(
            BRUTO["Company_Size_before_Layoffs"], method="spearman")), 3))

    def test_outras_agregacoes(self):
        y = BRUTO["Laid_Off"]
        self.assertEqual(consultar({"operacao": "mediana", "coluna": "Laid_Off"})["resultado"], float(y.median()))
        self.assertEqual(consultar({"operacao": "desvio_padrao", "coluna": "Laid_Off"})["resultado"], round(float(y.std()), 2))
        mx = consultar({"operacao": "maximo", "coluna": "Laid_Off"})
        self.assertEqual(mx["resultado"], float(y.max()))
        self.assertEqual(mx["eventos_correspondentes"][0]["Company"], BRUTO.loc[y.idxmax(), "Company"])

    def test_consulta_vazia(self):
        r = consultar({"operacao": "contagem", "filtros": [
            {"coluna": "Country", "operador": "eq", "valor": "Brazil"}, {"coluna": "year", "operador": "eq", "valor": 2021}]})
        self.assertEqual(int((BRUTO["Country"].eq("Brazil") & BRUTO["Year"].eq(2021)).sum()), 0)
        self.assertEqual(r["resultado"], 0)
        self.assertTrue(any("não significa que não houve demissões" in o for o in r["observacoes"]))
        r = consultar({"operacao": "soma", "coluna": "Laid_Off", "filtros": [
            {"coluna": "Country", "operador": "eq", "valor": "Brazil"}, {"coluna": "year", "operador": "eq", "valor": 2021}]})
        self.assertIsNone(r["resultado"])

    def test_operacoes_nao_permitidas(self):
        invalidas = [
            {"operacao": "apagar"},
            {"operacao": "soma", "coluna": "Percentage"},
            {"operacao": "media", "coluna": "__import__('os')"},
            {"operacao": "contagem", "filtros": [{"coluna": "Company", "operador": "gt", "valor": "A"}]},
            {"operacao": "contagem", "filtros": [{"coluna": "latitude", "operador": "eq", "valor": 1}]},
            {"operacao": "contagem", "filtros": [{"coluna": "year", "operador": "eq", "valor": "2023; DROP TABLE"}]},
            {"operacao": "contagem", "filtros": [{"coluna": "Date_layoffs", "operador": "eq", "valor": "ontem"}]},
            {"operacao": "ranking", "coluna": "Laid_Off"},
            {"operacao": "soma", "agrupar_por": ["latitude"]},
            {"operacao": "soma", "agrupar_por": ["year"], "limite": 1000},
            {"operacao": "contagem", "filtros": [{"coluna": "Country", "operador": "eq", "valor": "Wakanda"}]},
            "SELECT * FROM layoffs",
        ]
        for q in invalidas:
            with self.subTest(q=q):
                r = F.consultar_dados(q)
                json_estrito(r)
                self.assertIn("erro", r)

    def test_serializacao_json_estrita(self):
        consultas = [{"operacao": op, "coluna": "Money_Raised_clean", "agrupar_por": ["Stage"], "limite": 20}
                     for op in ("media", "mediana", "minimo", "maximo", "desvio_padrao")]
        for q in consultas:
            r = consultar(q)
            self.assertNotIn("erro", r)
            self.assertIn("(não informado)", [g["Stage"] for g in r["resultado"]])


class TestFatos(unittest.TestCase):
    def test_fatos_curtos_e_reais(self):
        fatos = F.fatos_do_dataset()
        self.assertGreater(len(fatos), 20)
        self.assertTrue(all(len(f) < 450 for f in fatos), "trechos curtos para o RAG")
        self.assertTrue(all("nan" not in f.lower().split() for f in fatos))
        total = f"{int(BRUTO['Laid_Off'].sum()):,}".replace(",", ".")
        self.assertTrue(any(total in f for f in fatos))
        for ano in range(2020, 2026):
            self.assertTrue(any(f"ano de {ano}" in f for f in fatos), ano)
        self.assertFalse(any("aluguel" in f.lower() for f in fatos))


if __name__ == "__main__":
    unittest.main()
