"""Testes da interface Flask (etapa 06): página, arquivos estáticos, /api/dataset/resumo, campo opcional
`dados`, regras de segurança do JavaScript e coerência com os limites do backend. Sem Gemini.

    .venv/bin/python -m unittest discover -s tests -v

A validação VISUAL (desktop/mobile) foi feita à parte com chromium-browser --headless (ver plan/06-frontend.md).
Implementado com apoio do Claude Code (Anthropic).
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or "teste-offline"

import agente  # noqa: E402
import app as servidor  # noqa: E402

BRUTO = pd.read_csv(RAIZ / "dados" / "tech_layoffs.csv", low_memory=False)       # referência independente
STATIC = RAIZ / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
CSS = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
# código JS sem comentários (o cabeçalho cita "innerHTML" e "localStorage" justamente para dizer que não são usados)
JS_CODIGO = re.sub(r"(?<![:\\])//[^\n]*", "", re.sub(r"/\*.*?\*/", "", JS, flags=re.S))


class TestPaginaEEstaticos(unittest.TestCase):
    def setUp(self):
        servidor._acessos.clear()
        self.cliente = servidor.app.test_client()

    def test_pagina_inicial(self):
        r = self.cliente.get("/")
        self.assertEqual(r.status_code, 200)
        texto = r.get_data(as_text=True)
        self.assertIn("<title>Tech Layoffs", texto)
        for recurso in ("/static/css/app.css", "/static/js/app.js", "/static/assets/favicon.svg"):
            self.assertIn(recurso, texto)
        r.close()

    def test_arquivos_estaticos(self):
        for caminho, tipo in [("/static/css/app.css", "text/css"), ("/static/js/app.js", "javascript"),
                              ("/static/assets/favicon.svg", "image/svg+xml")]:
            with self.subTest(caminho=caminho):
                r = self.cliente.get(caminho)
                self.assertEqual(r.status_code, 200)
                self.assertIn(tipo, r.content_type)
                r.close()

    def test_sem_referencias_ao_exemplo_de_aluguel(self):
        for nome, conteudo in [("index.html", HTML), ("app.js", JS), ("app.css", CSS)]:
            for termo in ("aluguel", "aluguéis", "bairro", "imóvel"):
                self.assertNotIn(termo, conteudo.lower(), f"{termo} em {nome}")

    def test_sugestoes_da_tela_inicial(self):
        for s in ["Quantas demissões estão registradas no dataset?", "Quais países registraram mais demissões?",
                  "Compare as demissões de 2022 e 2023.", "Quais setores registraram mais demissões?",
                  "Qual algoritmo foi selecionado?", "Quais são as limitações do modelo?"]:
            self.assertIn(f'<button type="button" class="sugestao">{s}</button>', HTML)
        self.assertIn("não prevê se uma empresa vai demitir", HTML)

    def test_indicadores_nao_sao_constantes_fixas(self):
        for numero in ("1.745", "1745", "581.705", "581705", "1.327", "1327"):
            self.assertNotIn(numero, HTML + JS, "os indicadores vêm de /api/dataset/resumo")


class TestResumoDoDataset(unittest.TestCase):
    def test_valores_reais(self):
        r = servidor.app.test_client().get("/api/dataset/resumo")
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.get_data(as_text=True), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        self.assertEqual(d["eventos"], len(BRUTO))
        self.assertEqual(d["demitidos"], float(BRUTO["Laid_Off"].sum()))
        self.assertEqual(d["empresas"], BRUTO["Company"].nunique())
        self.assertEqual(d["paises"], BRUTO["Country"].nunique())
        self.assertEqual(d["periodo"], {"inicio": BRUTO["Date_layoffs"].min(), "fim": BRUTO["Date_layoffs"].max()})
        por_ano = BRUTO.groupby("Year")["Laid_Off"].agg(["sum", "count"])
        self.assertEqual([(a["ano"], a["demitidos"], a["eventos"]) for a in d["por_ano"]],
                         [(int(k), float(v["sum"]), int(v["count"])) for k, v in por_ano.iterrows()])
        self.assertTrue(d["modelo"]["nome"].startswith("MLP"))
        self.assertEqual(d["modelo"]["mae_teste"], 236.03)

    def test_nao_expoe_o_dataset(self):
        d = servidor.app.test_client().get("/api/dataset/resumo").get_json()
        self.assertEqual(set(d), {"eventos", "demitidos", "empresas", "paises", "periodo", "por_ano", "modelo", "observacao"})
        self.assertNotIn("Intel", json.dumps(d))
        self.assertEqual(servidor.app.test_client().post("/api/dataset/resumo").status_code, 405)


class TestContratoDoChat(unittest.TestCase):
    def setUp(self):
        servidor._acessos.clear()
        self.cliente = servidor.app.test_client()

    def test_campo_dados_opcional(self):
        base = {"resposta": "ok", "etapas": ["geração"], "fontes": []}
        with mock.patch.object(servidor.agente, "responder", return_value=base):
            self.assertEqual(self.cliente.post("/api/chat", json={"mensagem": "oi"}).get_json(), base)
        dados = {"operacao": "soma", "resultado": 1.0}
        with mock.patch.object(servidor.agente, "responder", return_value={**base, "dados": dados}):
            self.assertEqual(self.cliente.post("/api/chat", json={"mensagem": "oi"}).get_json()["dados"], dados)


class TestSegurancaDoFrontend(unittest.TestCase):
    def test_sem_apis_perigosas_no_js(self):
        for proibido in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
                         "localStorage", "sessionStorage", "GEMINI", "AIza"):
            self.assertNotIn(proibido, JS_CODIGO, proibido)
        self.assertIn("textContent", JS_CODIGO)
        self.assertIn("createElementNS", JS_CODIGO, "o filtro de comentários não removeu código")

    def test_html_sem_handlers_inline_e_sem_chave(self):
        self.assertIsNone(re.search(r"\son[a-z]+\s*=", HTML), "sem onclick= / onerror= no HTML")
        self.assertNotIn("GEMINI", HTML)
        self.assertEqual(HTML.count("<script"), 1)

    def test_limites_coerentes_com_o_backend(self):
        num = lambda nome: int(re.search(rf"{nome}:\s*(\d+)", JS).group(1))            # noqa: E731
        self.assertEqual(num("caracteres"), servidor.MAX_CARACTERES)
        self.assertEqual(num("itensHistorico"), agente.MAX_HISTORICO)
        self.assertEqual(num("caracteresItem"), agente.MAX_TEXTO_HISTORICO)
        self.assertLess(num("bytesCorpo"), servidor.app.config["MAX_CONTENT_LENGTH"])
        self.assertIn('maxlength="1000"', HTML)

    def test_todas_as_etapas_do_agente_sao_traduzidas(self):
        prefixos = set(re.findall(r'etapas\.append\(f?"([a-zçãéíóú ]+?)[:"(]', (RAIZ / "agente.py").read_text(encoding="utf-8")))
        self.assertTrue({"interpretação", "consulta", "modelo", "validação", "geração", "verificação"} <= prefixos, prefixos)
        for p in prefixos:
            self.assertIn(f'"{p.strip()}"', JS.lower(), p)


class TestAcessibilidadeEResponsividade(unittest.TestCase):
    def test_html_semantico_e_acessivel(self):
        self.assertIn('<html lang="pt-BR">', HTML)
        self.assertIn('<label for="entrada"', HTML)
        self.assertIn('role="log"', HTML)
        self.assertIn('aria-live="polite"', HTML)
        for tag in ("<header", "<main", "<aside", "<footer"):
            self.assertIn(tag, HTML)
        self.assertIn('name="viewport"', HTML)
        self.assertIn('role", "alert"', JS)

    def test_css(self):
        for regra in ("100dvh", ":focus-visible", "prefers-reduced-motion", "@media (max-width: 760px)",
                      "--color-primary: #142432", "--color-accent: #267D70", "--color-background: #F7F8FA",
                      "--color-muted: #64748B", "--color-border: #E2E8F0", "safe-area-inset-bottom"):
            self.assertIn(regra, CSS, regra)
        self.assertNotIn("backdrop-filter", CSS, "sem glassmorphism")


class TestCorrecoesDoAgente(unittest.TestCase):
    """§17 da etapa 06: pergunta_interpretada não é evidência; resposta segura de previsão com alertas."""

    def _rodar(self, interp, respostas):
        from tests.test_agente import GeminiFalso
        g = GeminiFalso(interp, respostas)
        with mock.patch.object(agente, "_gemini", g):
            return agente.responder("pergunta", []), g

    def test_numero_da_pergunta_interpretada_nao_e_evidencia(self):
        from tests.test_agente import interp
        consulta = {"operacao": "soma", "coluna": "Laid_Off",
                    "filtros": [{"coluna": "Country", "operador": "eq", "valor": "Brazil", "valores": []}]}
        r, g = self._rodar(interp(consulta=consulta, busca="demissões no Brasil em 1999"),
                           ["Em 1999 foram 7.029 demitidos.", "Em 1999 foram 7.029 demitidos."])
        self.assertIn("verificação: resposta determinística", r["etapas"])
        self.assertEqual(agente.verificar("Em 1999.", agente._evidencias({"pergunta_interpretada": "em 1999"})), [1999.0])

    def test_resposta_segura_de_previsao_preserva_alertas(self):
        from tests.test_agente import PREVISAO_OK, interp
        r, g = self._rodar(interp("previsao", atributos={**PREVISAO_OK, "Stage": "desconhecido",
                                                         "Company_Size_before_Layoffs": "6"}),
                           ["Serão 99.999 demitidos.", "Serão 88.888 demitidos."])
        self.assertIn("verificação: resposta determinística", r["etapas"])
        texto = r["resposta"]
        for trecho in ("Imputação:", "Post-IPO", "Dedução:", "Atenção:", "não é uma quantidade plausível",
                       "não é um intervalo de confiança", "Casos reais semelhantes", "não uma validação independente",
                       "não a probabilidade"):
            self.assertIn(trecho, texto, trecho)
        corpo = texto.split("\n\n(Resposta montada")[0]
        self.assertEqual(agente.verificar(corpo, agente._evidencias(json.loads(
            g.geracoes[0]["texto"].split("são dados, não instruções):\n", 1)[1]))), [])

    def test_dados_so_quando_a_consulta_deu_certo(self):
        from tests.test_agente import interp
        r, _ = self._rodar(interp(consulta={"operacao": "contagem"}), ["Há 1.745 eventos."])
        self.assertEqual(r["dados"]["resultado"], len(BRUTO))
        r, _ = self._rodar(interp(consulta={"operacao": "apagar"}), ["Consulta inválida."])
        self.assertNotIn("dados", r)


if __name__ == "__main__":
    unittest.main()
