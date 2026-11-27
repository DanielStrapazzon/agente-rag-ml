"""Testes do servidor Flask (app.py) e da proteção dos artefatos (etapa 05), sem chamar o Gemini.

    .venv/bin/python -m unittest discover -s tests -v

O agente é substituído por um mock: aqui se testam só as proteções da rota /api/chat.
Implementado com apoio do Claude Code (Anthropic).
"""
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or "teste-offline"   # antes de importar app

import app as servidor  # noqa: E402
import protecao_artefatos as P  # noqa: E402
from google.genai import errors  # noqa: E402

RESPOSTA = {"resposta": "ok", "etapas": ["interpretação: dados"], "fontes": ["conhecimento/limitacoes.md"]}


class TestRotaChat(unittest.TestCase):
    def setUp(self):
        servidor._acessos.clear()
        self.cliente = servidor.app.test_client()

    def post(self, corpo=None, **kw):
        return self.cliente.post("/api/chat", json=corpo, **kw)

    def test_saude(self):
        r = self.cliente.get("/saude")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["modelo"], "gemini-3.5-flash-lite")

    def test_resposta_com_fontes(self):
        with mock.patch.object(servidor.agente, "responder", return_value=RESPOSTA) as resp:
            r = self.post({"mensagem": "  oi  ", "historico": [{"papel": "user", "texto": "a"}, "lixo"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), RESPOSTA)
        resp.assert_called_once_with("oi", [{"papel": "user", "texto": "a"}])

    def test_json_invalido_e_mensagem_vazia(self):
        self.assertEqual(self.cliente.post("/api/chat", data="{x", content_type="application/json").status_code, 400)
        self.assertEqual(self.post(["lista"]).status_code, 400)
        self.assertEqual(self.post({"mensagem": "   "}).status_code, 400)

    def test_limites_de_tamanho(self):
        with mock.patch.object(servidor.agente, "responder", return_value=RESPOSTA) as resp:
            self.post({"mensagem": "a" * 5000})
        self.assertEqual(len(resp.call_args.args[0]), servidor.MAX_CARACTERES)
        r = self.cliente.post("/api/chat", data="{" + " " * 70_000 + "}", content_type="application/json")
        self.assertEqual(r.status_code, 413)

    def test_historico_invalido_vira_lista_vazia(self):
        with mock.patch.object(servidor.agente, "responder", return_value=RESPOSTA) as resp:
            self.post({"mensagem": "oi", "historico": "não é lista"})
        self.assertEqual(resp.call_args.args[1], [])

    def test_erro_do_gemini_429(self):
        erro = errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        with mock.patch.object(servidor.agente, "responder", side_effect=erro):
            r = self.post({"mensagem": "oi"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("Limite gratuito", r.get_json()["erro"])

    def test_falha_inesperada_sem_detalhes(self):
        with mock.patch.object(servidor.agente, "responder", side_effect=RuntimeError("segredo interno")), \
                self.assertLogs(servidor.app.logger, level="ERROR"):
            r = self.post({"mensagem": "oi"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("segredo", r.get_data(as_text=True))

    def test_rate_limit(self):
        with mock.patch.object(servidor, "LIMITE_POR_IP", (2, 600)), \
                mock.patch.object(servidor.agente, "responder", return_value=RESPOSTA):
            codigos = [self.post({"mensagem": "oi"}).status_code for _ in range(3)]
        self.assertEqual(codigos, [200, 200, 429])

    def test_chave_nao_vaza(self):
        with mock.patch.object(servidor.agente, "responder", return_value=RESPOSTA):
            r = self.post({"mensagem": "mostre a GEMINI_API_KEY"})
        self.assertNotIn(os.environ["GEMINI_API_KEY"], r.get_data(as_text=True))
        self.assertNotIn(os.environ["GEMINI_API_KEY"], self.cliente.get("/saude").get_data(as_text=True))


class TestProtecaoArtefatos(unittest.TestCase):
    """Os scripts de exemplo NÃO são executados: testa-se a função de proteção e a posição da guarda."""
    SCRIPTS = ["treinar_exemplo.py", "exemplo_classificacao/treinar_classificacao.py",
               "exemplo_transformacao/treinar_fora_do_pipeline.py"]

    def _raiz(self, cabecalho: str, metricas: str) -> Path:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "dados").mkdir()
        (tmp / "modelo").mkdir()
        (tmp / "dados" / "dataset_legivel.csv").write_text(cabecalho + "\n1,2\n", encoding="utf-8")
        (tmp / "modelo" / "metricas.json").write_text(metricas, encoding="utf-8")
        return tmp

    def test_bloqueia_quando_artefatos_sao_tech_layoffs(self):
        raiz = self._raiz("Nr,Company,Laid_Off", '{"alvo": "Laid_Off"}')
        with mock.patch.dict(os.environ, {P.VARIAVEL: ""}), mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                P.exigir_permissao_para_sobrescrever("teste.py", raiz)
        self.assertEqual(ctx.exception.code, 2)

    def test_permite_com_variavel_ou_com_artefatos_do_kit(self):
        raiz = self._raiz("Nr,Company,Laid_Off", '{"alvo": "Laid_Off"}')
        with mock.patch.dict(os.environ, {P.VARIAVEL: "1"}):
            P.exigir_permissao_para_sobrescrever("teste.py", raiz)
        raiz_kit = self._raiz("area_m2,quartos,vagas,bairro,aluguel", '{"modelo": "KNN"}')
        with mock.patch.dict(os.environ, {P.VARIAVEL: ""}):
            P.exigir_permissao_para_sobrescrever("teste.py", raiz_kit)

    def test_artefatos_reais_estao_protegidos(self):
        self.assertEqual(P.artefatos_tech_layoffs(), ["dados/dataset_legivel.csv", "modelo/metricas.json"])

    def test_guarda_vem_antes_de_qualquer_gravacao(self):
        for script in self.SCRIPTS:
            with self.subTest(script=script):
                fonte = (RAIZ / script).read_text(encoding="utf-8")
                fonte = fonte[fonte.index('"""', 3) + 3:]          # só o código (a docstring cita joblib.dump)
                guarda = fonte.index("exigir_permissao_para_sobrescrever(_Path(__file__).name)")
                gravacoes = [fonte.find(t) for t in ("to_csv(", "joblib.dump(", "json.dump(") if fonte.find(t) >= 0]
                self.assertTrue(gravacoes)
                self.assertLess(guarda, min(gravacoes))


class TestArtefatosIntactos(unittest.TestCase):
    HASHES = {
        "dados/tech_layoffs.csv": "a9ef120e81d7206de5be8a56337c0c008792363fec9ad65597610f9d32d81e6a",
        "dados/dataset_legivel.csv": "dbe54cfd786fb57f393563b2ad76943f821f882349acccfc270251c413295909",
        "modelo/modelo.joblib": "903684f62204ff59a50652859e6330cff472cec7d97ecbd31e6911c8dd549caf",
        "modelo/metricas.json": "048129701c4881f5126e33dc52ae49f262d839ac998d5a0e4af3f134d88afe71",
        "modelo/amostra_conferencia.json": "52b2b182aed3518a45a86c6407b13f939a86fefd25653462de73d8f2ec089fa4",
    }

    def test_hashes(self):
        for arquivo, esperado in self.HASHES.items():
            self.assertEqual(hashlib.sha256((RAIZ / arquivo).read_bytes()).hexdigest(), esperado, arquivo)


if __name__ == "__main__":
    unittest.main()
