# Perfil do dataset Tech Layoffs

> Gerado na etapa 02 a partir do processamento real de `dados/tech_layoffs.csv` com
> `preparar_dados.py` (pandas 3.0.5, Python 3.14.10). Todos os números abaixo foram calculados do CSV;
> nada foi estimado. Quando algo é interpretação e não fato, está indicado.

## 1. Domínio

Cada linha é um **evento de demissão** registrado em uma empresa de tecnologia (ou com atuação em
tecnologia), com a data, o número de pessoas demitidas, o porte da empresa antes e depois, a
localização da sede, o setor, o estágio de financiamento e o capital captado.

Três quantidades diferentes, que não devem ser confundidas:

| Medida | Valor | O que é |
|---|---|---|
| **Eventos** | **1.745** | linhas do dataset (anúncios de demissão) |
| **Empresas distintas** | **1.327** (1.323 ignorando maiúsculas/minúsculas) | uma empresa pode ter vários eventos |
| **Funcionários demitidos** | **581.705** | soma de `Laid_Off` em todos os eventos |

## 2. Origem

- Arquivo `tech_layoffs.csv` enviado pelo estudante no trabalho anterior (ver
  `reports/README_TRABALHO_ANTERIOR.md`).
- Página relacionada citada: https://www.kaggle.com/datasets/ulrikeherold/tech-layoffs-2020-2024
- **Versão exata no Kaggle: não confirmada.** O título da página fala em 2020–2024, mas o CSV tem
  registros até 2025-12-28; não há evidência no repositório de qual versão foi baixada.

## 3. Período

- `Date_layoffs`: de **2020-03-12** a **2025-12-28**; todas as 1.745 datas são válidas (ISO `AAAA-MM-DD`).
- 2021 tem apenas 14 eventos — muito menos que os anos vizinhos. Não há como saber, só pelo CSV, se isso
  reflete a realidade ou lacunas na coleta.

## 4. Registros antes e depois da limpeza

| Passo | Linhas |
|---|---|
| CSV original | 1.745 × 18 colunas |
| Duplicatas completas removidas | 0 |
| Removidas por alvo ausente, não numérico ou negativo | 0 |
| Datas inválidas (mantidas como ausentes) | 0 |
| **Dataset preparado** | **1.745 × 21 colunas** (18 originais + 3 derivadas) |

A limpeza não removeu nenhuma linha neste CSV; as regras existem para manter a metodologia e
proteger contra versões futuras do arquivo.

## 5. Dicionário dos atributos

Unidades marcadas com "(inferida)" vêm do nome da coluna ou da relação entre colunas, não de uma
documentação oficial.

| Coluna | Tipo | Unidade | Ausentes | Únicos | Uso no modelo anterior |
|---|---|---|---|---|---|
| `Nr` | inteiro | identificador (1–2412, com lacunas) | 0 | 1.745 | não usado |
| `Company` | texto | nome da empresa | 0 | 1.327 | não usado |
| `Location_HQ` | texto | cidade da sede | 0 | 181 | não usado |
| `Region` | texto | região (`other`, `San Francisco Bay Area`, `Cascadia`, `Tokyo`) | 0 | 4 | não usado |
| `USState` | texto | estado/província da sede; `non` em 214 linhas fora dos EUA | 1 | 66 | não usado |
| `Country` | texto | país da sede | 0 | 42 | **entrada categórica** |
| `Continent` | texto | continente da sede | 0 | 6 | **entrada categórica** |
| `Laid_Off` | inteiro | **pessoas demitidas no evento** | 0 | 316 | **alvo** |
| `Date_layoffs` | data | AAAA-MM-DD | 0 | 741 | origem de `month` |
| `Percentage` | decimal | % do quadro demitido (inferida) | 0 | 81 | **excluída: vazamento** |
| `Company_Size_before_Layoffs` | inteiro | funcionários antes do evento | 0 | 713 | **entrada numérica** |
| `Company_Size_after_layoffs` | inteiro | funcionários depois do evento | 0 | 886 | **excluída: vazamento** |
| `Industry` | texto | setor | 0 | 60 | **entrada categórica** |
| `Stage` | texto | estágio de financiamento (Seed, Series A…J, Post-IPO, Acquired…) | 27 | 16 | **entrada categórica** |
| `Money_Raised_in__mil` | decimal | capital captado, em milhões (inferida pelo nome; moeda não documentada, provavelmente US$) | 110 | 556 | origem de `Money_Raised_clean` |
| `Year` | inteiro | ano do evento | 0 | 6 | origem de `year` |
| `latitude`, `longitude` | decimal | graus, coordenadas da cidade-sede | 0 | 181 | não usados |
| `Money_Raised_clean` | decimal | **derivada**: igual a `Money_Raised_in__mil` convertido para número | 110 | — | **entrada numérica** |
| `year` | inteiro | **derivada**: `Year` convertido para número | 0 | — | **entrada numérica** |
| `month` | inteiro | **derivada**: mês de `Date_layoffs` (1–12) | 0 | — | **entrada numérica** |

Valores ausentes no total: `Money_Raised_in__mil`/`Money_Raised_clean` 110 (6,3%), `Stage` 27 (1,5%),
`USState` 1 (Sapiens, Israel). Nenhuma outra coluna tem ausentes. `Stage` também tem 228 linhas com o
texto `Unknown`, que **não** é contado como ausente.

## 6. Transformações realizadas (`preparar_dados.py`)

Reproduzem a célula 7 (treino) de `notebooks/analise_modelos.ipynb`:

1. nomes de colunas sem espaços nas pontas;
2. remoção de linhas duplicadas idênticas;
3. `Laid_Off` convertido para número; mantidas só as linhas com alvo presente e ≥ 0;
4. colunas numéricas convertidas com `pd.to_numeric(errors="coerce")` (texto inválido → ausente);
5. `Date_layoffs` convertido para data (data inválida → ausente, linha mantida);
6. derivadas `Money_Raised_clean`, `year` (de `Year`) e `month` (de `Date_layoffs`).

**Não** aplicadas (pertencem ao Pipeline e serão ajustadas só no treino): imputação, `StandardScaler`,
`OneHotEncoder`, `LabelEncoder`, `log1p` do alvo. Nenhum ausente foi trocado por zero e nenhuma grafia
de categoria ou empresa foi alterada.

Exportação (`dados/dataset_legivel.csv` desde a etapa 04; até a etapa 03, `dados/dataset_legivel_tech_layoffs.csv`): UTF-8, separador vírgula, quebra `\n`, sem
índice; data `AAAA-MM-DD`; contagens inteiras escritas sem `.0` (`20` em vez de `20.0`); ausentes
como campo vazio; colunas originais na ordem do CSV seguidas de `Money_Raised_clean`, `year`, `month`.

## 7. Estatísticas descritivas

### Alvo `Laid_Off` (pessoas demitidas por evento)

| Estatística | Valor |
|---|---|
| Eventos | 1.745 |
| Total de demitidos | 581.705 |
| Média por evento | 333,36 |
| Mediana por evento | 90 |
| Desvio-padrão | 1.188,00 |
| Mínimo / máximo | 3 / 22.000 |
| Quartis (25% / 75%) | 40 / 200 |
| Percentis 90% / 95% / 99% | 500 / 1.000 / 5.824 |
| Assimetria (skewness) | 9,55 |
| Eventos com mais de 1.000 demitidos | 81 |
| Eventos com 10.000 ou mais | 9 |
| Outliers pela regra 1,5 × IQR (> 440) | 216 |

A média é quase 4 vezes a mediana: poucos eventos muito grandes puxam a média. Isso explica o uso
de `log1p` no alvo no trabalho anterior.

Maiores eventos:

| Empresa | Data | Demitidos | País | Setor |
|---|---|---|---|---|
| Intel | 2025-04-23 | 22.000 | USA | Hardware |
| Intel | 2024-08-01 | 15.000 | USA | Hardware |
| Tesla | 2024-04-15 | 14.000 | USA | Motor Vehicle Manufacturing |
| Google | 2023-01-20 | 12.000 | USA | Consumer |
| TCS | 2025-07-27 | 12.000 | India | IT Services and IT Consulting |

### Por ano (coluna `Year`)

| Ano | Eventos | Empresas distintas | Demitidos | % do total | Média/evento | Mediana |
|---|---|---|---|---|---|---|
| 2020 | 330 | 320 | 61.529 | 10,6% | 186,5 | 63 |
| 2021 | 14 | 14 | 6.790 | 1,2% | 485,0 | 170 |
| 2022 | 568 | 513 | 127.577 | 21,9% | 224,6 | 84,5 |
| 2023 | 474 | 432 | 176.946 | 30,4% | 373,3 | 100 |
| 2024 | 265 | 246 | 122.661 | 21,1% | 462,9 | 100 |
| 2025 | 94 | 90 | 86.202 | 14,8% | 917,0 | 150 |

2022 tem mais eventos; 2023 tem mais demitidos.

### Por país (10 maiores em demitidos; 42 países no total)

| País | Eventos | Demitidos | % do total | Média/evento | Mediana |
|---|---|---|---|---|---|
| USA | 1.121 | 425.063 | 73,1% | 379,2 | 100 |
| India | 123 | 41.492 | 7,1% | 337,3 | 150 |
| Germany | 59 | 25.809 | 4,4% | 437,4 | 75 |
| Sweden | 18 | 17.057 | 2,9% | 947,6 | 125 |
| UK | 55 | 13.409 | 2,3% | 243,8 | 80 |
| Israel | 86 | 11.985 | 2,1% | 139,4 | 35 |
| Canada | 89 | 10.815 | 1,9% | 121,5 | 50 |
| Brazil | 48 | 7.029 | 1,2% | 146,4 | 100 |
| Singapore | 23 | 5.187 | 0,9% | 225,5 | 80 |
| Japan | 2 | 5.120 | 0,9% | 2.560,0 | 2.560 |

25 dos 42 países têm menos de 5 eventos.

### Por continente

| Continente | Eventos | Demitidos | % do total | Média/evento | Mediana |
|---|---|---|---|---|---|
| North America | 1.213 | 436.108 | 75,0% | 359,5 | 90 |
| Asia | 257 | 69.583 | 12,0% | 270,8 | 100 |
| Europe | 176 | 62.102 | 10,7% | 352,9 | 89,5 |
| South America | 56 | 7.801 | 1,3% | 139,3 | 100 |
| Oceana | 30 | 3.689 | 0,6% | 123,0 | 60 |
| Africa | 13 | 2.422 | 0,4% | 186,3 | 54 |

### Por setor (10 maiores em demitidos; 60 setores no total)

| Setor | Eventos | Demitidos | % do total | Média/evento | Mediana |
|---|---|---|---|---|---|
| Consumer | 90 | 61.155 | 10,5% | 679,5 | 140 |
| Retail | 140 | 59.228 | 10,2% | 423,1 | 100 |
| Hardware | 14 | 57.000 | 9,8% | 4.071,4 | 1.700 |
| Other | 102 | 56.721 | 9,8% | 556,1 | 117 |
| Transportation | 109 | 39.046 | 6,7% | 358,2 | 140 |
| Finance | 227 | 36.185 | 6,2% | 159,4 | 71 |
| Food | 102 | 33.702 | 5,8% | 330,4 | 130,5 |
| Software Development | 11 | 17.115 | 2,9% | 1.555,9 | 100 |
| Infrastructure | 25 | 16.798 | 2,9% | 671,9 | 80 |
| Healthcare | 123 | 16.109 | 2,8% | 131,0 | 90 |

Finance é o setor com mais eventos (227), mas Consumer, Retail e Hardware somam mais demitidos.
30 dos 60 setores têm menos de 5 eventos.

### Por estágio (`Stage`)

| Estágio | Eventos | Demitidos | % do total | Média/evento | Mediana |
|---|---|---|---|---|---|
| Post-IPO | 417 | 346.123 | 59,5% | 830,0 | 221 |
| Unknown | 228 | 57.005 | 9,8% | 250,0 | 70 |
| Acquired | 113 | 31.062 | 5,3% | 274,9 | 130 |
| Series B | 220 | 24.403 | 4,2% | 110,9 | 47 |
| (ausente) | 27 | 24.137 | 4,1% | 894,0 | 170 |
| Series D | 177 | 20.958 | 3,6% | 118,4 | 87 |
| Series C | 188 | 18.030 | 3,1% | 95,9 | 50 |
| Series E | 107 | 17.208 | 3,0% | 160,8 | 100 |
| Series F | 55 | 9.357 | 1,6% | 170,1 | 120 |
| Private Equity | 29 | 8.398 | 1,4% | 289,6 | 106 |
| Series H | 23 | 7.131 | 1,2% | 310,0 | 150 |
| Series A | 99 | 4.906 | 0,8% | 49,6 | 29 |
| Series G | 16 | 4.594 | 0,8% | 287,1 | 175 |
| Series I | 8 | 3.155 | 0,5% | 394,4 | 325 |
| Series J | 5 | 2.350 | 0,4% | 470,0 | 180 |
| Subsidiary | 5 | 1.688 | 0,3% | 337,6 | 209 |
| Seed | 28 | 1.200 | 0,2% | 42,9 | 18,5 |

### Demais numéricas

| Coluna | Não nulos | Mínimo | Mediana | Média | Máximo |
|---|---|---|---|---|---|
| `Company_Size_before_Layoffs` | 1.745 | 4 | 575 | 4.540,43 | 600.000 |
| `Company_Size_after_layoffs` | 1.745 | 0 | 453 | 4.207,08 | 588.000 |
| `Percentage` | 1.745 | 0 | 15 | 22,35 | 100 |
| `Money_Raised_clean` | 1.635 | 1 | 174 | 736,10 | 121.900 |

Eventos por mês (`month`): jan 170, fev 133, mar 146, **abr 248**, mai 185, jun 208, jul 135,
ago 123, set 73, out 93, nov 142, dez 89.

Correlação com `Laid_Off` (Spearman, por postos, menos sensível aos outliers):
`Company_Size_before_Layoffs` 0,798; `Money_Raised_clean` 0,414; `year` 0,154; `month` −0,029.
Pearson: 0,694; 0,108; 0,118; −0,053. Correlação não implica causa.

## 8. Inconsistências identificadas (mantidas, não corrigidas)

| Tipo | Detalhe | Linhas |
|---|---|---|
| `Year` ≠ ano de `Date_layoffs` | Acko (2020-04-01), Ada Support (2020-04-24), Abra (2022-06-30), Acast (2022-09-15), 99 (2022-09-20), Ada (2022-09-20), Absci (2023-09-05): todas com `Year` = 2024 | 7 |
| País com duas grafias | `UK` (55) e `United Kingdom` (3) | 58 |
| Grafias de país/continente | `Uruquay` (2), `United Arabian Emirates` (5), `Oceana` (30) | — |
| Setores quase duplicados | `Transportion` (3) × `Transportation` (109); `Logistic` (1) × `Logistics` (34); `Higher Education Industry` (1) × `Higher Education` (1); `Semiconductors` (1) × `Semiconductor Manufacturing` (1); `IT Services and IT Consulting` (1) × `IT Services` (2); `Gaming` (2) × `Computer Games` (2); `Finance` (227) × `Financial Services` (1) | — |
| Setor com texto de descrição | `The Animation Tool for UI/UX Designers.` | 1 |
| Empresa com grafias diferentes | `Freshbooks`/`FreshBooks`, `Clearco`/`ClearCo`, `7shifts`/`7Shifts`, `UiPath`/`UIPath` | 1.327 → 1.323 empresas |
| `Percentage` arredondado a 0 | Amazon: 150 de 333.483 (≈ 0,045%) | 1 |
| `Percentage` diverge do cálculo | Arrival: 480 de 2.080 (≈ 23%), registrado 30% | 1 |
| Estágio desconhecido | `Unknown` (texto) além de 27 ausentes | 228 |

Consistências verificadas: `Company_Size_before_Layoffs − Laid_Off = Company_Size_after_layoffs`
com diferença ≤ 1 em todas as linhas; `Laid_Off` nunca supera o porte anterior; cada país pertence a
um único continente; `USState = non` só aparece fora dos EUA; 77 eventos com 100% do quadro
(porte depois = 0).

As grafias foram **mantidas** porque o modelo do trabalho anterior foi treinado com elas;
unificá-las mudaria a metodologia e deve ser uma decisão explícita na etapa de ML.

## 9. Diferenças entre o dataset original e o preparado

| | `dados/tech_layoffs.csv` | `dados/dataset_legivel.csv` |
|---|---|---|
| Linhas | 1.745 | 1.745 (nenhuma removida) |
| Colunas | 18 | 21 (+ `Money_Raised_clean`, `year`, `month`) |
| Contagens | `20.0`, `50.0`… | `20`, `50`… (mesmo valor) |
| Data | `2020-03-12` | `2020-03-12` (validada) |
| Categorias e empresas | originais | idênticas |
| Ausentes | 138 células | idênticos + 110 em `Money_Raised_clean` |
| Valores numéricos | — | idênticos (testado com tolerância zero) |

`Money_Raised_clean` = `Money_Raised_in__mil` e `year` = `Year` em todas as linhas deste CSV; as
colunas existem para corresponder 1:1 às entradas do notebook.

## 10. Divergência entre EDA e treino no notebook

A EDA (célula 5) limpava o capital com regex (`[^0-9.\-]` removido antes de converter) e o treino
(célula 7) usava `pd.to_numeric` direto. Neste CSV, os 1.635 valores e os 110 ausentes são
**idênticos** nas duas formas (muda só o tipo: `Float64` × `float64`). Foi adotada a forma do
**treino**, que gerou as métricas registradas. O regex também transformaria notação científica
(`1e3`) em outro número (`13`).

## 11. Limitações

- O modelo do trabalho anterior estima o tamanho de eventos **já registrados**; não estima a chance de uma empresa demitir.
- Só há eventos que viraram notícia/registro; empresas sem demissões não aparecem (viés de seleção).
- Distribuição do alvo muito assimétrica (assimetria 9,55; 216 outliers pelo IQR): erros absolutos são grandes nos eventos grandes.
- Muitas categorias raras (30 setores e 25 países com menos de 5 eventos) e grafias duplicadas.
- 298 empresas aparecem em mais de um evento (até 6); uma divisão aleatória treino/teste pode colocar eventos da mesma empresa nos dois lados.
- 2021 tem só 14 eventos; 2025 tem 94, com média por evento bem mais alta (917,0).
- `Company_Size_before_Layoffs` é a variável mais associada ao alvo, mas sua disponibilidade antes do anúncio deve ser justificada.
- Versão da fonte no Kaggle não confirmada; unidade/moeda de `Money_Raised_in__mil` não documentada.
