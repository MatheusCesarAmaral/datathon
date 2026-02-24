# datathon

Aplicação Streamlit para predição de risco educacional com entrada manual e análise em lote via planilha.

## Estrutura

- `scripts/app.py`: aplicação Streamlit
- `scripts/modelo_risco.pkl`: modelo treinado (joblib)
- `scripts/analise_datathon.ipynb`: notebook de exploração/treinamento
- `data/`: arquivos de dados e materiais do desafio

## Requisitos

- Python 3.11 (recomendado)

## Instalação

```bash
pip install -r scripts/requirements.txt
```

## Executar a aplicação

A partir da raiz do repositório:

```bash
streamlit run scripts/app.py
```

## Uso da planilha

1. Baixe o modelo de planilha (`.xlsx`) pelo botão na aplicação.
2. Preencha as colunas obrigatórias:
   - `IDA`, `IEG`, `IPV`, `Matem`, `Portug`, `Inglês`, `IAA`, `IPS`, `IAN`, `Defas`
3. Faça upload da planilha `.xlsx`.

### Regras de validação

- `IDA`, `IEG`, `IPV`, `Matem`, `Portug`, `Inglês`, `IAA`, `IPS`: valores entre `0` e `10`
- `IAN`, `Defas`: valores entre `-10` e `10`
- Valores vazios ou não numéricos são rejeitados

## Notebook

O notebook `scripts/analise_datathon.ipynb` foi ajustado para usar caminho relativo da base:

- `../data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`
