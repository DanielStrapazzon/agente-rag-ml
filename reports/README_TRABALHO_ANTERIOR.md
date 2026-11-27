# UNOESC — Trabalho de ML com Tech Layoffs

O projeto **já inclui o CSV real enviado pelo aluno**: `data/tech_layoffs.csv` (1.745 linhas / 18 colunas). Não é mock.

## Google Colab
1. Extraia o ZIP no computador. Envie a pasta **Trabalho_Tech_Layoffs_DADOS_REAIS** inteira para a raiz de **Meu Drive**. Não renomeie a pasta sem mudar a variável ROOT na primeira célula.
2. Abra `notebooks/analise_modelos.ipynb` com Google Colab.
3. Execute **Ambiente de execução > Executar tudo** e autorize a montagem do Google Drive.
4. O notebook lê `data/tech_layoffs.csv` e salva figuras e tabelas em `reports/resultados/`.
5. Confira métricas, discuta resultados e complete o relatório no template oficial da UNOESC (template não fornecido). Entregue notebook executado + PDF.

## Requisitos acadêmicos
Trilha de regressão (Linear, KNN, MLP), ≥ 1000 linhas, ≥ 8 atributos, variáveis categóricas, EDA (heatmap/boxplot/histograma/frequências), imputação, encoding, StandardScaler dentro de Pipeline, train/test fixo, GridSearchCV 5-fold, R²/MAE/RMSE, previsto × real, resíduos, comparação e limitações.

## Fonte e integridade
O CSV foi enviado pelo estudante, identificado como `tech_layoffs.csv`. Página relacionada: https://www.kaggle.com/datasets/ulrikeherold/tech-layoffs-2020-2024 . **Antes de entregar, confirme que o link representa exatamente a versão utilizada e publique no fórum da disciplina.** Notebook EDA de referência em `referencias/`.

## Interpretação
Variável-alvo: `Laid_Off` (número de demitidos por evento registrado). Não usar `Percentage` e `Company_Size_after_layoffs` como preditores: vazamento de resposta. Modelo não prevê ocorrência de demissões futuras. `Company_Size_before_Layoffs` deve ser conhecido no momento da previsão.

## Relatório
Use `reports/ROTEIRO_RELATORIO.md` e o template oficial. Inclua a seção 'Declaração de uso de IA' e esteja preparado para explicar todas as células.
