# Exemplo: transformações FORA do Pipeline

Use este exemplo se, no trabalho anterior, vocês transformaram os dados **antes** de treinar, com `pd.get_dummies`, `LabelEncoder`, `StandardScaler` aplicado direto no DataFrame, mapas de códigos (1 = masculino, 2 = feminino) etc.

## A regra
O agente trabalha com o **dataset legível**: textos por extenso e números na escala original. É ele que vai para o RAG, para as estatísticas e para os casos semelhantes, porque as pessoas perguntam "e no Centro?", e não "bairro_3 = 1". Já o **modelo** só entende os dados **transformados**. Então, antes de cada previsão, a entrada do usuário precisa passar **pelas mesmas transformações do treino**, com os **mesmos objetos ajustados no treino** (o scaler do treino, a mesma lista de colunas, os mesmos mapas). **Não reajustem nada na hora da previsão.**

## Passo a passo (na pasta do kit)
```bash
python exemplo_transformacao/treinar_fora_do_pipeline.py
```
Depois, em `ferramentas.py`, troquem a função `para_o_modelo()` pela versão em `exemplo_transformacao/para_o_modelo_fora_do_pipeline.py`, incluindo a linha `TRANSF = joblib.load(...)`, e rodem:
```bash
python testar_sem_api.py        # o item "Conferência da ponte" tem que dar ok: True
```

## Como saber se esqueci alguma transformação
O treino salva `modelo/amostra_conferencia.json`: 5 entradas legíveis e a previsão que o modelo deu no treino. Ao iniciar, o app refaz essas 5 previsões passando por `para_o_modelo()`. Se alguma transformação faltar ou estiver diferente (escala, ordem das colunas, código de categoria), as previsões não batem, e aparece no log:
`ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino`.

## Se o CSV do trabalho anterior já estava transformado
Por exemplo, um CSV com categorias em códigos e números já padronizados. Nesse caso, gerem o `dataset_legivel.csv` a partir do CSV **original** do Kaggle, só com a limpeza (sem codificar nem escalar). O dicionário de códigos (ex.: 1 = masculino) também entra na base de conhecimento.
