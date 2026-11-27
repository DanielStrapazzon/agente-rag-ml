# Descrição do dataset Tech Layoffs

## Objetivo do dataset
O dataset Tech Layoffs reúne eventos de demissão (layoffs) em empresas de tecnologia. Cada linha é um evento: uma empresa anunciou que demitiu um certo número de funcionários numa data.

O projeto usa o dataset para análise estatística das demissões no setor de tecnologia e para um modelo de regressão que estima quantos funcionários foram demitidos num evento com determinadas características.

## Origem dos dados
O arquivo tech_layoffs.csv foi enviado pelo estudante no trabalho anterior. A página relacionada citada é o dataset "Tech Layoffs 2020-2024" de ulrikeherold no Kaggle.

A versão exata baixada do Kaggle não está confirmada: o título da página fala em 2020 a 2024, mas o CSV tem registros até dezembro de 2025.

## Período dos registros
As datas dos eventos (Date_layoffs) vão de 2020-03-12 a 2025-12-28. Todas as 1.745 datas são válidas, no formato AAAA-MM-DD.

O ano de 2021 tem só 14 eventos, bem menos que os anos vizinhos. Só pelo CSV não dá para saber se isso reflete a realidade ou falhas na coleta.

## Eventos, empresas e funcionários demitidos
Eventos, empresas e funcionários são três medidas diferentes. O dataset tem 1.745 eventos de demissão (linhas), 1.327 empresas distintas e 581.705 funcionários demitidos no total (soma de Laid_Off).

Uma mesma empresa pode ter vários eventos: 298 empresas aparecem mais de uma vez, até 6 vezes. Ignorando maiúsculas e minúsculas no nome, são 1.323 empresas, porque algumas aparecem com grafias diferentes (Freshbooks e FreshBooks, Clearco e ClearCo, 7shifts e 7Shifts, UiPath e UIPath).

## Colunas de identificação e localização
- Nr: identificador do evento, sem uso no modelo.
- Company: nome da empresa (company).
- Location_HQ: cidade da sede (headquarters), 181 cidades.
- Region: região da sede, com 4 valores: other, San Francisco Bay Area, Cascadia e Tokyo.
- USState: estado ou província da sede; usa o texto "non" em 214 eventos fora dos Estados Unidos e está ausente em 1 evento.
- Country: país da sede (país), 42 países, com nomes em inglês (USA = Estados Unidos, UK = Reino Unido, Brazil = Brasil).
- Continent: continente da sede, 6 valores: North America, Asia, Europe, South America, Oceana (grafia do dataset para Oceania) e Africa.
- latitude e longitude: coordenadas da cidade-sede, em graus.

## Colunas do evento de demissão
- Laid_Off: número de funcionários demitidos (laid off) no evento. É a variável-alvo do modelo.
- Date_layoffs: data do evento de demissão.
- Year: ano do evento. Em 7 eventos, Year é diferente do ano de Date_layoffs.
- Percentage: percentual do quadro de funcionários demitido no evento (unidade inferida pelo nome e pela relação entre as colunas).
- Company_Size_before_Layoffs: número de funcionários da empresa antes da demissão (porte anterior).
- Company_Size_after_layoffs: número de funcionários depois da demissão (porte posterior).

## Colunas da empresa
- Industry: setor de atuação (setor, indústria), 60 valores, como Finance (finanças), Retail (varejo), Healthcare (saúde) e Transportation (transporte).
- Stage: estágio de financiamento, como Seed (semente), Series A a Series J, Post-IPO (pós-IPO, capital aberto), Acquired (adquirida), Private Equity, Subsidiary e Unknown (desconhecido). Está ausente em 27 eventos.
- Money_Raised_in__mil: capital captado pela empresa (money raised), em milhões. A moeda não está documentada no dataset; provavelmente é dólar americano, mas isso não está confirmado. Está ausente em 110 eventos.

## Atributos derivados
O projeto cria três atributos, calculados como no notebook do trabalho anterior:
- Money_Raised_clean: o capital captado convertido para número; tem os mesmos valores de Money_Raised_in__mil.
- year: o ano do evento, tirado da coluna Year (e não da data).
- month: o mês do evento (1 a 12), tirado de Date_layoffs.

## Unidades dos atributos
Laid_Off, Company_Size_before_Layoffs e Company_Size_after_layoffs são contagens de pessoas (funcionários). Percentage é um percentual de 0 a 100. Money_Raised_clean está em milhões, sem moeda confirmada. year e month são o ano e o mês do evento.

## Atributos usados no modelo
O modelo de Machine Learning usa 8 entradas. Categóricas: Country (país), Continent (continente), Industry (setor) e Stage (estágio). Numéricas: Company_Size_before_Layoffs (porte anterior), Money_Raised_clean (capital captado), year (ano) e month (mês). A variável-alvo é Laid_Off (número de demitidos).

## Atributos excluídos do modelo
Percentage e Company_Size_after_layoffs ficaram fora do modelo porque são calculadas a partir da própria demissão. Usá-las seria vazamento da resposta (data leakage): Company_Size_before_Layoffs menos Laid_Off é igual a Company_Size_after_layoffs, com diferença de no máximo 1, em todos os eventos.

Nr, Company, Location_HQ, Region, USState, latitude e longitude também não são entradas do modelo, mas podem ser usadas em consultas históricas.

## Preparação dos dados
O script preparar_dados.py repete a limpeza do notebook: remove linhas duplicadas, mantém só eventos com Laid_Off numérico e maior ou igual a zero, converte números e datas e cria os atributos derivados. Nenhuma linha foi removida: o dataset preparado tem 1.745 eventos e 21 colunas.

Valores ausentes foram mantidos (não viraram zero), e os nomes de países, setores e empresas foram preservados exatamente como no arquivo original.
