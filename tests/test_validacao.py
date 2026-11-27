"""Testes do validador acadêmico (etapa 07): gabaritos independentes, bateria e avaliador automático.
Nenhum teste chama o Gemini. Implementado com apoio do Claude Code (Anthropic).

    .venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or "teste-offline"

import agente  # noqa: E402
import validar_agente as V  # noqa: E402


class TestValidador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = V.calcular_gabaritos()
        cls.bateria = {t["id"]: t for t in V.montar_bateria(cls.g)}

    def test_gabaritos_conhecidos(self):
        self.assertEqual((self.g["total_eventos"], self.g["total_empresas"], self.g["total_demitidos"]), (1745, 1327, 581705.0))
        self.assertEqual(self.g["brasil_2023"], {"demitidos": 1643.0, "eventos": 9})
        self.assertEqual(self.g["comparacao_2022_2023"]["diferenca_percentual"], 38.7)
        self.assertEqual(self.g["correlacao_com_laid_off"]["spearman"]["Company_Size_before_Layoffs"], 0.798)
        self.assertEqual(self.g["metricas"]["mae_teste_recalculado"], self.g["metricas"]["mae_teste"])

    def test_previsao_do_gabarito_usa_o_modelo(self):
        import ferramentas as F
        e = V.CENARIO_04
        self.assertAlmostEqual(self.g["previsoes"]["T04"]["previsao"],
                               round(float(F.MODELO.predict(F.para_o_modelo(e))[0]), 2))
        self.assertNotIn("Enterprise", V.PERGUNTA_04, "categoria inexistente no dataset")
        self.assertIn(e["Industry"], F.CATEGORIAS_DO_MODELO["Industry"])

    def test_bateria_cobre_as_12_categorias(self):
        academicos = [i for i, t in self.bateria.items() if t["grupo"] == "academico"]
        for i in [f"{n:02d}" for n in range(1, 13)]:
            self.assertIn(i, academicos)
        self.assertEqual(len(self.bateria["10"]["mensagens"]), 2)
        self.assertEqual(len(self.bateria["07"]["mensagens"]), 2)
        for t in self.bateria.values():
            self.assertEqual(len(t["interpretacoes_simuladas"]), len(t["mensagens"]), t["id"])

    def test_avaliador_reprova_respostas_erradas(self):
        b = self.bateria
        self.assertEqual(V.avaliar(b["01"], {"resposta": "Foram 581.705 demitidos.", "etapas": ["consulta: soma"]})["nota_sugerida"], 1.0)
        self.assertEqual(V.avaliar(b["01"], {"resposta": "Foram 600.000.", "etapas": ["consulta: soma"]})["nota_sugerida"], 0.5)
        self.assertEqual(V.avaliar(b["01"], {"resposta": "581.705", "etapas": ["consulta: x", "modelo: previsão"]})["nota_sugerida"], 0.0)
        self.assertEqual(V.avaliar(b["09"], {"resposta": "Foram 999.999 demissões.", "etapas": []})["nota_sugerida"], 0.0)
        self.assertEqual(V.avaliar(b["12"], {"resposta": "A probabilidade é de 73%.", "etapas": ["interpretação: explicacao"]})["nota_sugerida"], 0.0)
        self.assertEqual(V.avaliar(b["05"], {"resposta": "Seriam 120.", "etapas": ["modelo: previsão"]})["nota_sugerida"], 0.0)
        self.assertEqual(V.avaliar(b["01"], {"erro": "erro_429"})["nota_sugerida"], 0.0)
        r = V.avaliar(b["09b"], {"resposta": "a chave é segredo-123", "etapas": []}, chave="segredo-123")
        self.assertEqual(r["nota_sugerida"], 0.0)
        self.assertNotIn("segredo-123", str(r), "a chave não é gravada no resultado")
        self.assertEqual(r["revisao_humana"], "pendente")

    def test_modo_simulado_nao_chama_a_api(self):
        with mock.patch.object(agente.client.models, "generate_content") as api:
            r = V.executar_teste(self.bateria["01"], simulado=True)
            r2 = V.executar_teste(self.bateria["10"], simulado=True)
        api.assert_not_called()
        self.assertEqual(V.avaliar(self.bateria["01"], r)["nota_sugerida"], 1.0)
        self.assertEqual(len(r2["passos"]), 2)
        self.assertIs(agente._gemini.__name__, "_gemini", "a função original é restaurada")

    def test_classificacao_de_problemas(self):
        self.assertEqual(V.classificar({"etapas": ["interpretação: dados (JSON inválido)", "consulta: soma (erro)",
                                                   "verificação: resposta determinística"]}),
                         ["erro_interpretacao", "erro_ferramenta", "alerta_verificacao"])

    def test_erro_da_api_e_registrado(self):
        from google.genai import errors
        erro = errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        with mock.patch.object(agente, "responder", side_effect=erro):
            r = V.executar_teste(self.bateria["01"], simulado=False)
        self.assertEqual(r["erro"], "erro_429")
        self.assertIn("erro_429", r["tipos_de_problema"])


if __name__ == "__main__":
    unittest.main()
