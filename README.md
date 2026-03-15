# Datathon - Risco Educacional

Aplicacao e notebook para analise de risco educacional a partir da base PEDE, com consolidacao dos anos `2022`, `2023` e `2024`.

## O que o projeto entrega

- Notebook de analise e treinamento do modelo.
- Aplicacao Streamlit para previsao individual e em lote.
- Modelo final treinado com as mesmas variaveis exigidas no app.

## Estrutura

- `scripts/analise_datathon.ipynb`: notebook principal de exploracao, analise e treinamento.
- `scripts/preparar_base_pede.py`: leitura e padronizacao das abas `PEDE2022`, `PEDE2023` e `PEDE2024`.
- `scripts/retrain_model.py`: retreino do modelo final com as 10 features do app.
- `scripts/app.py`: aplicação Streamlit.
- `scripts/modelo_risco.pkl`: modelo treinado usado pela aplicacao.
- `scripts/requirements.txt`: dependencias do projeto.
- `data/BASE DE DADOS PEDE - DATATHON.xlsx`: base original com 3 abas.
- `data/`: materiais de apoio do desafio.

## Base de dados

O arquivo `data/BASE DE DADOS PEDE - DATATHON.xlsx` contem tres abas:

- `PEDE2022`
- `PEDE2023`
- `PEDE2024`

O script `scripts/preparar_base_pede.py`:

- le as 3 abas;
- padroniza nomes de colunas entre os anos;
- cria a base unica usada no notebook e no retreino;
- gera colunas auxiliares como `Ano Referencia`, `Pedra Atual` e `INDE Atual`.

## Modelo final

Para manter consistencia entre analise e produto, o modelo final foi treinado apenas com as variaveis disponiveis no Streamlit:

- `IDA`
- `IEG`
- `IPV`
- `Matem`
- `Portug`
- `Ingles`
- `IAA`
- `IPS`
- `IAN`
- `Defas`

Isso evita desalinhamento entre o que o modelo consome no treino e o que a aplicacao pede do usuario.

## Requisitos

- Python `3.11+`
- Ambiente virtual recomendado

## Instalação

Na raiz do projeto:

```bash
python -m venv .venv
```

Ative o ambiente virtual.

PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Depois instale as dependencias:

```bash
pip install -r scripts/requirements.txt
```

## Como executar o notebook

Abra o arquivo `scripts/analise_datathon.ipynb` no Jupyter/VS Code e rode as celulas em ordem.

O notebook cobre:

- consolidacao da base `2022-2024`;
- criacao da variavel de risco;
- treinamento e avaliacao do modelo;
- analise de importancia das features;
- graficos para responder perguntas do PDF, incluindo:
  - `IAA` vs `IDA`
  - `IAA` vs `IEG`
  - correlacao entre indicadores
  - comparacao por ano
  - distribuicao de risco por ano e por fase

## Como retreinar o modelo

Depois de ativar o ambiente:

```bash
.venv\Scripts\python.exe scripts/retrain_model.py
```

Esse script:

- carrega a base unificada;
- usa apenas as 10 features do app;
- treina um `RandomForestClassifier`;
- imprime o `classification_report`;
- salva o resultado em `scripts/modelo_risco.pkl`.

## Como executar o app

Na raiz do projeto:

```bash
streamlit run scripts/app.py
```

## Uso da aplicacao

### Entrada manual

A aplicacao aceita as seguintes variaveis:

- `IDA`
- `IEG`
- `IPV`
- `Matem`
- `Portug`
- `Ingles`
- `IAA`
- `IPS`
- `IAN`
- `Defas`

### Upload de planilha

1. Baixe a planilha modelo no proprio app.
2. Preencha as 10 colunas obrigatórias.
3. Envie a planilha `.xlsx`.
4. Baixe o resultado com a probabilidade e o nivel de risco.

## Regras de validacao no app

- `IDA`, `IEG`, `IPV`, `Matem`, `Portug`, `Ingles`, `IAA`, `IPS`: valores entre `0` e `10`
- `IAN`, `Defas`: valores entre `-10` e `10`
- valores vazios ou nao numericos sao rejeitados

## Observacoes importantes

- A base consolidada tem dados dos anos `2022`, `2023` e `2024`.
- O modelo final esta alinhado com o app.
- A variavel `Ingles` possui muitos valores ausentes na base original, entao a amostra efetiva do treino e menor que o total de linhas consolidadas.
