from pathlib import Path

import pandas as pd


ARQUIVO_PADRAO = (
    Path(__file__).resolve().parents[1] / "data" / "BASE DE DADOS PEDE - DATATHON.xlsx"
)


CONFIG_ABAS = {
    "PEDE2022": {
        "ano_referencia": 2022,
        "pedra_atual": "Pedra 22",
        "inde_atual": "INDE 22",
        "renomear": {
            "Nome": "Nome Aluno",
            "Ano nasc": "Data de Nasc",
            "Idade 22": "Idade",
            "Fase ideal": "Fase ideal",
            "Defas": "Defas",
        },
    },
    "PEDE2023": {
        "ano_referencia": 2023,
        "pedra_atual": "Pedra 2023",
        "inde_atual": "INDE 2023",
        "renomear": {
            "Nome Anonimizado": "Nome Aluno",
            "Mat": "Matem",
            "Por": "Portug",
            "Ing": "Inglês",
            "Fase Ideal": "Fase ideal",
            "Defasagem": "Defas",
        },
    },
    "PEDE2024": {
        "ano_referencia": 2024,
        "pedra_atual": "Pedra 2024",
        "inde_atual": "INDE 2024",
        "renomear": {
            "Nome Anonimizado": "Nome Aluno",
            "Mat": "Matem",
            "Por": "Portug",
            "Ing": "Inglês",
            "Fase Ideal": "Fase ideal",
            "Defasagem": "Defas",
        },
    },
}


def carregar_base_unificada(caminho: Path | str = ARQUIVO_PADRAO) -> pd.DataFrame:
    caminho = Path(caminho)
    frames = []

    for aba, config in CONFIG_ABAS.items():
        df_ano = pd.read_excel(caminho, sheet_name=aba).copy()
        df_ano.columns = [str(col).strip() for col in df_ano.columns]
        df_ano = df_ano.rename(columns=config["renomear"])

        df_ano["Ano Referencia"] = config["ano_referencia"]
        df_ano["Pedra Atual"] = df_ano.get(config["pedra_atual"])
        df_ano["INDE Atual"] = df_ano.get(config["inde_atual"])

        frames.append(df_ano)

    colunas_unificadas = sorted(set().union(*(frame.columns for frame in frames)))
    frames = [frame.reindex(columns=colunas_unificadas) for frame in frames]

    base = pd.concat(frames, ignore_index=True, sort=False)
    base = base.loc[:, ~base.columns.duplicated()].copy()

    return base
