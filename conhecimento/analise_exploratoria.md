# Base de conhecimento: análise do trabalho anterior (EXEMPLO — substituam pelo texto de vocês)

## Sobre o dataset
O dataset reúne anúncios de aluguel de imóveis residenciais (dados SINTÉTICOS do kit). Cada linha é um imóvel, com área em metros quadrados, número de quartos, número de vagas de garagem, bairro e o valor do aluguel mensal em reais.

## Descrição das colunas
- area_m2: área útil do imóvel, em metros quadrados.
- quartos: número de quartos (1 a 4).
- vagas: número de vagas de garagem (0 a 2).
- bairro: um de cinco bairros de Chapecó (Centro, Efapi, Passo dos Fortes, Presidente Médici, Jardim Itália).
- aluguel: valor do aluguel mensal em reais (alvo do modelo).

## Principais achados da análise exploratória
- A área é a variável que mais influencia o aluguel: quanto maior o imóvel, maior o valor.
- O bairro tem efeito forte: Centro e Jardim Itália são os mais caros; Efapi e Presidente Médici, os mais baratos.
- Quartos e vagas têm efeito menor quando a área já é conhecida.
- Não há valores faltantes; alguns aluguéis muito baixos para imóveis grandes foram mantidos (ruído do dataset).

## Sobre o modelo
O modelo escolhido foi o KNN com 7 vizinhos e pesos por distância, dentro de um Pipeline com StandardScaler nas colunas numéricas e OneHotEncoder no bairro. Ele foi comparado com Regressão Linear e MLP no GridSearchCV com validação cruzada de 5 dobras.
O KNN estima o aluguel olhando os imóveis mais parecidos do treino. Por isso, ele não extrapola bem: para imóveis fora da faixa de área vista no treino, a previsão tende a ficar presa nos valores extremos.

## Limitações
- O modelo não conhece bairros fora da lista.
- Os dados são de um único momento; preços mudam com o tempo.
- A margem de erro típica (MAE) deve sempre ser informada junto com a previsão.
