# Base de conhecimento: análise do trabalho anterior (EXEMPLO DE CLASSIFICAÇÃO — dados sintéticos)

## Sobre o dataset
Clientes de cartão de crédito com limite, idade, meses com atraso nos últimos 6 meses, renda mensal e escolaridade. O alvo indica se o cliente ficou inadimplente ("sim" ou "nao").

## Principais achados da análise exploratória
- O número de meses com atraso é a variável mais associada à inadimplência: quem atrasou mais vezes tem risco bem maior.
- Limite de crédito e renda mais altos estão associados a menor risco.
- As classes são desbalanceadas (cerca de 1 inadimplente para cada 3 adimplentes), por isso a acurácia sozinha engana; o F1 da classe "sim" é a métrica principal.

## Sobre o modelo
MLPClassifier dentro de um Pipeline com StandardScaler e OneHotEncoder, escolhido no GridSearchCV com validação cruzada de 5 dobras (métrica F1 macro).

## Limitações
- O modelo erra mais na classe minoritária (inadimplentes).
- É uma estimativa sobre um dataset público, não uma decisão de crédito.
