# Análise exploratória (EDA) do Tech Layoffs

## Gráficos da análise exploratória
O trabalho anterior gerou com Seaborn um heatmap de correlação, um boxplot e um histograma do número de demissões, e gráficos de frequência por setor (Industry), país (Country) e ano (Year). Os arquivos estão em reports/resultados/.

## Distribuição das demissões por evento
O número de demitidos por evento (Laid_Off) vai de 3 a 22.000. A média é 333,36 e a mediana é 90 funcionários por evento. Metade dos eventos tem 90 demitidos ou menos.

Os quartis de Laid_Off são 40 (25%) e 200 (75%). Os percentis 90, 95 e 99 são 500, 1.000 e 5.824 demitidos.

## Assimetria e outliers
A distribuição das demissões é muito assimétrica (skewness de 9,55): há muitos eventos pequenos e poucos eventos gigantes. A média é quase 4 vezes a mediana porque esses poucos eventos grandes puxam a média para cima.

Pela regra de 1,5 vez o intervalo interquartil (IQR), 216 eventos são outliers (mais de 440 demitidos). 81 eventos têm mais de 1.000 demitidos e 9 têm 10.000 ou mais.

Por causa da assimetria, o modelo do trabalho anterior foi treinado com o logaritmo do número de demitidos (transformação log1p do alvo).

## Maiores eventos de demissão
Os maiores eventos do dataset são: Intel com 22.000 demitidos em 2025-04-23, Intel com 15.000 em 2024-08-01, Tesla com 14.000 em 2024-04-15, Google com 12.000 em 2023-01-20 e TCS com 12.000 em 2025-07-27.

## Concentração geográfica por país
As demissões estão muito concentradas nos Estados Unidos (USA): 1.121 eventos e 425.063 demitidos, 73,1% do total. Em seguida vêm Índia (India, 41.492 demitidos, 7,1%), Alemanha (Germany, 25.809, 4,4%), Suécia (Sweden, 17.057, 2,9%) e Reino Unido grafado como UK (13.409, 2,3%).

Muitos países têm poucos registros: 25 dos 42 países têm menos de 5 eventos. O Japão (Japan) tem só 2 eventos, mas 5.120 demitidos, o que dá média de 2.560 por evento.

## Concentração por continente
Por continente, a América do Norte (North America) tem 1.213 eventos e 75,0% dos demitidos. Depois vêm Ásia (Asia, 12,0%), Europa (Europe, 10,7%), América do Sul (South America, 1,3%), Oceania (Oceana, 0,6%) e África (Africa, 0,4%).

## Comportamento histórico por ano
2022 é o ano com mais eventos de demissão (568), mas 2023 é o ano com mais funcionários demitidos (176.946, 30,4% do total).

2020 teve 330 eventos e 61.529 demitidos. 2021 teve só 14 eventos e 6.790 demitidos. 2024 teve 265 eventos e 122.661 demitidos. 2025 teve 94 eventos e 86.202 demitidos.

A média de demitidos por evento cresceu nos últimos anos: 224,6 em 2022, 373,3 em 2023, 462,9 em 2024 e 917,0 em 2025. Isso descreve os eventos registrados; não é uma previsão para os próximos anos.

## Comparação entre setores
Finance (finanças) é o setor com mais eventos de demissão (227), mas não é o que tem mais demitidos (36.185, média 159,4 por evento).

Os setores com mais funcionários demitidos são Consumer (61.155), Retail (varejo, 59.228), Hardware (57.000) e Other (56.721). Hardware tem só 14 eventos, mas média de 4.071,4 demitidos por evento, puxada pelos eventos da Intel.

Healthcare (saúde) tem 123 eventos e 16.109 demitidos, com média de 131,0 por evento. 30 dos 60 setores têm menos de 5 eventos.

## Estágio de financiamento
Empresas de capital aberto (Post-IPO) concentram 59,5% dos demitidos: 417 eventos e 346.123 demitidos, com média de 830,0 por evento. Empresas em estágios iniciais demitem menos por evento: Seed tem média de 42,9 e Series A, de 49,6.

## Correlações observadas
A variável mais associada ao número de demitidos é o porte anterior da empresa (Company_Size_before_Layoffs), com correlação de Spearman de 0,798 (Pearson 0,694). Empresas maiores tendem a ter eventos com mais demitidos.

O capital captado (Money_Raised_clean) tem correlação de Spearman de 0,414 com Laid_Off (Pearson 0,108). O ano tem 0,154 e o mês, −0,029: praticamente nenhuma associação.

Correlação não é causalidade. Uma correlação alta mostra que duas variáveis variam juntas no dataset, não que uma provoca a outra.

## Eventos por mês
Abril é o mês com mais eventos (248), seguido de junho (208) e maio (185). Setembro tem menos eventos (73).
