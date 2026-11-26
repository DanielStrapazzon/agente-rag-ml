# Exemplo de CLASSIFICAÇÃO

Para quem fez a trilha de **classificação** no trabalho anterior. Na pasta do kit:

```bash
python exemplo_classificacao/treinar_classificacao.py      # gera dados/ e modelo/ de classificação
```
Depois, copiem os arquivos (no Windows: `copy` no lugar de `cp`, `del` no lugar de `rm`, e `\` no lugar de `/`):
```bash
cp exemplo_classificacao/ferramentas_classificacao.py ferramentas.py
rm conhecimento/analise_exploratoria.md
cp exemplo_classificacao/analise_classificacao.md conhecimento/
python testar_sem_api.py
python conferir.py
```
Por fim, troquem a persona em `prompts/responder.md` ("Assistente de Crédito", classe + probabilidade em vez de valor + MAE) e o título da página.

O que muda em relação à regressão:
- `prever()` devolve a classe (`previsao`) e a probabilidade de cada classe (modelos SVC precisam de `probability=True` no treino), mais a acurácia e o F1 de teste. O `metricas.json` precisa ter `acuracia_teste` e `f1_teste`.
- `casos_semelhantes()` conta as classes reais dos vizinhos e diz se a previsão concorda com a maioria.
- `estatisticas("<alvo>")` devolve a contagem e a proporção de cada classe (4 casas). `estatisticas(col, "<alvo>")` devolve as médias por classe.
- `fatos_do_dataset()` gera frases com as proporções por grupo e as médias por classe, para o RAG.
