# Limitações dos dados e do modelo

## Viés de seleção
O dataset só tem eventos de demissão que foram registrados ou noticiados. Eventos não divulgados ou não coletados não aparecem, então os números descrevem a base de dados, não necessariamente todas as demissões do setor de tecnologia.

## Ausência de empresas sem demissões
Empresas que não demitiram não estão no dataset. Por isso não dá para comparar empresas que demitiram com empresas que não demitiram.

## O modelo não estima probabilidade de demissão
O modelo estima quantos funcionários foram demitidos num evento com certas características. Ele não estima a probabilidade (chance, risco) de uma empresa fazer demissões, nem prevê quando vai acontecer uma demissão.

## Ausência de registros não é ausência de demissões
Quando uma consulta não encontra eventos (por exemplo, um país num ano), isso só quer dizer que o dataset não tem registros com esses filtros. Não prova que não houve demissões no mundo real.

## Divisão aleatória entre treino e teste
A divisão entre treino e teste foi aleatória, e não em ordem cronológica. Por isso as métricas não mostram como o modelo se sairia em anos futuros.

## Empresas repetidas entre treino e teste
121 das 327 empresas do conjunto de teste também aparecem no treino (131 eventos de teste). O modelo pode ter visto eventos parecidos da mesma empresa, o que tende a deixar as métricas otimistas.

## Categorias raras
30 dos 60 setores (Industry) e 25 dos 42 países (Country) têm menos de 5 eventos. Para essas categorias o modelo aprendeu com muito poucos exemplos.

No teste há categorias que não existem no treino: o país South Korea (Coreia do Sul) e os setores Appliances, Financial Services, Logistic, Motor Vehicle Manufacturing, Semiconductor Manufacturing e "The Animation Tool for UI/UX Designers.". O modelo ignora essas categorias.

## Grafias diferentes da mesma categoria
O dataset tem grafias diferentes para o mesmo valor: UK (55 eventos) e United Kingdom (3), Transportion (3) e Transportation (109), Logistic (1) e Logistics (34). As grafias foram mantidas como no original. O modelo trata cada grafia como uma categoria separada; as consultas estatísticas do agente agrupam as grafias equivalentes e avisam quando fazem isso.

## Eventos extremos
A distribuição dos demitidos é muito assimétrica: mediana 90, máximo 22.000. O modelo erra muito mais nos eventos grandes: MAE de 2.171,4 nos 18 eventos de teste com mais de 1.000 demitidos, contra 130,8 nos demais.

## Previsões acima do porte da empresa
Em 14 dos 349 eventos de teste, todos de empresas com até 120 funcionários, o modelo estimou mais demitidos do que a empresa tinha de funcionários. Isso não foi escondido nem corrigido: quando acontece, o agente avisa que a estimativa não é uma quantidade plausível.

## Divergências temporais
Em 7 eventos, a coluna Year é diferente do ano de Date_layoffs (Acko, Ada Support, Abra, Acast, 99, Ada e Absci; todos com Year = 2024). As análises anuais e o modelo usam year, derivado de Year, como no trabalho anterior. Filtros por data exata usam Date_layoffs.

## Origem dos dados
A versão exata do dataset no Kaggle não foi confirmada. A página citada fala em 2020 a 2024, mas o arquivo tem eventos até 2025-12-28.

## Moeda do capital captado
O capital captado (Money_Raised_in__mil e Money_Raised_clean) está em milhões, mas a moeda não está documentada. Provavelmente é dólar americano, mas isso não está confirmado.

## Disponibilidade do porte anterior
O porte anterior da empresa (Company_Size_before_Layoffs) é a entrada mais associada ao número de demitidos. Num uso real, esse número precisa ser conhecido antes do anúncio da demissão; senão a estimativa perde sentido.

## Estágio e capital desconhecidos
Quando o estágio (Stage) ou o capital captado é informado como desconhecido, o modelo preenche o valor do treino: o estágio mais frequente (Post-IPO) ou a mediana do capital (178 milhões). Isso pode mudar bastante a estimativa. O dataset também tem a categoria Unknown para estágio desconhecido, que pode ser informada explicitamente.

## Casos semelhantes não são validação independente
Os casos reais semelhantes mostrados junto com a estimativa vêm do mesmo dataset usado para desenvolver o modelo. A comparação ajuda a ver se a estimativa faz sentido, mas não prova que o modelo acertou.

## Limitações das métricas
O MAE é o erro absoluto médio observado no teste, não um intervalo de confiança. O R² mede a parte da variação explicada pelo modelo, não a porcentagem de acertos.
