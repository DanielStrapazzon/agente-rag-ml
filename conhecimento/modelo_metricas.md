# Modelo de Machine Learning e métricas

## O que o modelo faz
O modelo é uma regressão: ele estima quantos funcionários foram demitidos (Laid_Off) num evento de demissão com as características informadas (país, continente, setor, estágio, porte anterior, capital captado, ano e mês).

O modelo não prevê se uma empresa vai demitir, nem a probabilidade de acontecer uma demissão. O dataset só tem eventos de demissão que aconteceram.

## Modelos avaliados
O trabalho anterior comparou três algoritmos de regressão: Regressão Linear Múltipla, KNN (K-vizinhos mais próximos, KNeighborsRegressor) e MLP (rede neural perceptron multicamadas, MLPRegressor).

## Metodologia de treinamento
Os dados foram divididos aleatoriamente em 80% para treino (1.396 eventos) e 20% para teste (349 eventos), com random_state 42. Os hiperparâmetros foram escolhidos com GridSearchCV e validação cruzada (cross-validation) de 5 dobras (KFold com embaralhamento), usando o erro absoluto médio (MAE) como critério.

O novo treinamento (treinar.py, Python 3.14.10, scikit-learn 1.9.1) reproduziu a metodologia do notebook e obteve as mesmas métricas do trabalho anterior, com diferenças só de arredondamento (cerca de 10 elevado a −13).

## Pré-processamento
O pré-processamento fica dentro do Pipeline do modelo e é ajustado só nos dados de treino. Nas variáveis numéricas, os valores ausentes são preenchidos pela mediana e os valores são padronizados (StandardScaler). Nas categóricas, os ausentes recebem a categoria mais frequente e as categorias viram colunas 0/1 (OneHotEncoder); categorias desconhecidas são ignoradas.

O número de demitidos é transformado com log1p antes do treino e volta à escala original com expm1 (TransformedTargetRegressor), por causa da forte assimetria. Previsões negativas são trocadas por zero.

## Grades de hiperparâmetros
O KNN testou n_neighbors 3, 7 e 15, weights uniform e distance, e metric euclidean e manhattan. A MLP testou hidden_layer_sizes (32,) e (64, 32), activation relu e tanh, e alpha 0,001 e 0,1, com max_iter 400 e early_stopping. A Regressão Linear foi usada como referência (baseline), sem grade.

## Melhores parâmetros
Melhor KNN: 7 vizinhos (n_neighbors 7), pesos por distância (weights distance) e distância euclidiana. Melhor MLP: ativação tanh, alpha 0,1 e duas camadas ocultas de 64 e 32 neurônios.

## Métricas de cada modelo
MLP: MAE de validação cruzada 201,19; MAE no teste 236,03; RMSE no teste 851,53; R² no teste 0,5537.

KNN: MAE de validação cruzada 210,19; MAE no teste 258,29; RMSE no teste 884,34; R² no teste 0,5186.

Regressão Linear Múltipla: MAE de validação cruzada 101.025,85; MAE no teste 884,24; RMSE no teste 8.331,71; R² no teste −41,7277. O R² negativo vem da combinação de modelo linear em escala log com expm1, que em alguns eventos gera números enormes de demitidos.

## Seleção do modelo MLP
O modelo escolhido foi o MLP, por ter o menor MAE na validação cruzada (201,19 contra 210,19 do KNN). O conjunto de teste não foi usado para escolher. O MLP também tem o menor MAE no teste, então os dois critérios concordam.

A vantagem do MLP sobre o KNN é pequena: 9,0 no MAE de validação cruzada, 22,3 no MAE de teste e 0,035 no R². Com uma só divisão treino/teste, não dá para afirmar que o MLP é melhor de forma robusta.

## O que é o MAE (explicação para leigos)
O MAE (erro absoluto médio, mean absolute error) é a diferença média, em número de funcionários, entre o que o modelo estimou e o que aconteceu, medida nos 349 eventos de teste. O MAE do MLP é 236,03 funcionários.

O MAE não é um intervalo de confiança. Dizer "previsão mais ou menos o MAE" é só uma referência ilustrativa, sem garantia de que o valor real esteja nessa faixa.

## O que é o R² (explicação para leigos)
O R² (coeficiente de determinação) mostra quanto da variação do número de demitidos o modelo consegue explicar. O R² do MLP no teste é 0,5537: o modelo explica cerca de 55% da variação.

O R² não é uma porcentagem de acertos. Um R² de 0,5537 não quer dizer que o modelo acerta 55% das vezes.

## O que é o RMSE (explicação para leigos)
O RMSE (raiz do erro quadrático médio) também mede o erro em número de funcionários, mas pesa mais os erros grandes. O RMSE do MLP (851,53) é bem maior que o MAE (236,03) porque o modelo erra muito em alguns eventos gigantes.

## Onde o modelo erra mais
No teste, o erro absoluto mediano do MLP é 45,1 funcionários. O MAE é 2.171,4 nos 18 eventos com mais de 1.000 demitidos e 130,8 nos outros 331 eventos. O modelo é bem menos preciso nos eventos grandes.

## O modelo é confiável?
O modelo serve como estimativa estatística aproximada, não como certeza. Ele explica cerca de 55% da variação no teste (R² 0,5537), tem erro médio de 236,03 funcionários (MAE) e erra mais em eventos grandes. Deve ser usado junto com a comparação com casos reais semelhantes e com as limitações do dataset.
