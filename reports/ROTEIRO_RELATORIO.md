# Roteiro de relatório técnico — preenchimento após execução REAL

1. **Introdução:** problema — estimar número de demissões em eventos conhecidos, não prever ocorrência.
2. **Dados:** Kaggle, fonte e versão verificadas, 1.745 observações, 18 colunas originais, atributos categóricos, dados ausentes e seleção de preditores.
3. **EDA:** inserir e explicar heatmap, boxplot e histograma. Avaliar outliers e setores/países mais frequentes.
4. **Metodologia:** 80% treino/20% teste, SEED 42; pipeline com imputação, codificação e StandardScaler; 5-fold GridSearchCV; transformação log1p do alvo; grade de cada método.
5. **Resultados:** copiar `reports/resultados/metricas_teste.csv`, `melhores_parametros.csv`, gráficos de previstos x reais e resíduos. Interpretar R², MAE e RMSE em contexto.
6. **Comparação e discussão:** justificar vencedor; falar de erros altos em grandes demissões, possibilidades de viés, limitações, disponibilidade de atributos e divisão temporal futura.
7. **Conclusão:** preencher após observar os resultados, sem inventar métricas.
8. **Declaração de uso de IA:** descrever apoio do ChatGPT na estruturação e revisão, o que os integrantes executaram/validaram.

**Importante:** o modelo diagramado oficial da UNOESC não estava anexado; transcreva o conteúdo para ele antes da entrega.
