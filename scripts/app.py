from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("modelo_risco.pkl")
TEMPLATE_PLANILHA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=xlsx"
)
COLUNAS_ENTRADA = [
    "IDA",
    "IEG",
    "IPV",
    "Matem",
    "Portug",
    "Inglês",
    "IAA",
    "IPS",
    "IAN",
    "Defas",
]
LIMITES_COLUNAS = {
    "IDA": (0.0, 10.0),
    "IEG": (0.0, 10.0),
    "IPV": (0.0, 10.0),
    "Matem": (0.0, 10.0),
    "Portug": (0.0, 10.0),
    "Inglês": (0.0, 10.0),
    "IAA": (0.0, 10.0),
    "IPS": (0.0, 10.0),
    "IAN": (-10.0, 10.0),
    "Defas": (-10.0, 10.0),
}

st.set_page_config(page_title="Risco Educacional - Passos Mágicos", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def classificar_risco(probabilidade: float) -> str:
    if probabilidade < 0.30:
        return "Baixo"
    if probabilidade < 0.60:
        return "Moderado"
    if probabilidade < 0.80:
        return "Alto"
    return "Crítico"


def preparar_dataframe_para_modelo(df_usuario: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Monta um DataFrame com as colunas esperadas pelo modelo."""
    df_modelo = pd.DataFrame(0, index=df_usuario.index, columns=feature_names)

    for col in df_usuario.columns:
        if col in df_modelo.columns:
            df_modelo[col] = df_usuario[col]

    return df_modelo.fillna(0)


def validar_e_normalizar_planilha(df: pd.DataFrame) -> pd.DataFrame:
    faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    if faltando:
        st.error("A planilha está inválida.")
        st.write("Colunas obrigatórias faltando:")
        st.write(faltando)
        st.stop()

    df_validado = df.copy()

    for coluna in COLUNAS_ENTRADA:
        serie_original = df_validado[coluna]
        serie_numerica = pd.to_numeric(serie_original, errors="coerce")

        linhas_nao_numericas = df_validado.index[
            serie_numerica.isna() & serie_original.notna()
        ].tolist()
        if linhas_nao_numericas:
            st.error(f"A coluna '{coluna}' deve conter apenas números.")
            st.write(
                "Linhas com valores inválidos (1-based, até 10):",
                [i + 2 for i in linhas_nao_numericas[:10]],
            )
            st.stop()

        linhas_vazias = df_validado.index[serie_numerica.isna()].tolist()
        if linhas_vazias:
            st.error(f"A coluna '{coluna}' possui valores vazios.")
            st.write(
                "Linhas com valores ausentes (1-based, até 10):",
                [i + 2 for i in linhas_vazias[:10]],
            )
            st.stop()

        minimo, maximo = LIMITES_COLUNAS[coluna]
        fora_intervalo = df_validado.index[
            (serie_numerica < minimo) | (serie_numerica > maximo)
        ].tolist()
        if fora_intervalo:
            st.error(
                f"A coluna '{coluna}' deve estar no intervalo de {minimo} a {maximo}."
            )
            st.write(
                "Linhas fora do intervalo (1-based, até 10):",
                [i + 2 for i in fora_intervalo[:10]],
            )
            st.stop()

        df_validado[coluna] = serie_numerica.astype(float)

    return df_validado


try:
    model = load_model()
except Exception as exc:
    st.error("Não foi possível carregar o modelo de risco.")
    st.code(f"Arquivo esperado: {MODEL_PATH}")
    st.exception(exc)
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("O modelo carregado não possui 'feature_names_in_'.")
    st.stop()

feature_names = list(model.feature_names_in_)

if not hasattr(model, "predict_proba"):
    st.error("O modelo carregado não suporta 'predict_proba'.")
    st.stop()

# ---------------- TÍTULO ----------------

st.title("📊 Predição de Risco Educacional — Passos Mágicos")
st.caption("Sistema de alerta precoce para identificação de alunos em risco educacional.")

# ---------------- PREVISÃO INDIVIDUAL ----------------

st.header("Análise individual do aluno")

with st.sidebar:
    st.header("Entrada manual")

    IDA = st.number_input("IDA", 0.0, 10.0, 6.0, 0.1)
    IEG = st.number_input("IEG", 0.0, 10.0, 6.0, 0.1)
    IPV = st.number_input("IPV", 0.0, 10.0, 6.0, 0.1)
    Matem = st.number_input("Matem", 0.0, 10.0, 6.0, 0.1)
    Portug = st.number_input("Portug", 0.0, 10.0, 6.0, 0.1)
    Ingles = st.number_input("Inglês", 0.0, 10.0, 6.0, 0.1)
    IAA = st.number_input("IAA", 0.0, 10.0, 6.0, 0.1)
    IPS = st.number_input("IPS", 0.0, 10.0, 6.0, 0.1)
    IAN = st.number_input("IAN", -10.0, 10.0, 0.0, 1.0)
    Defas = st.number_input("Defas", -10.0, 10.0, 0.0, 1.0)

    st.divider()
    analisar = st.button("🔮 Calcular risco")

if analisar:
    entrada = pd.DataFrame(
        [[IDA, IEG, IPV, Matem, Portug, Ingles, IAA, IPS, IAN, Defas]],
        columns=COLUNAS_ENTRADA,
    )

    df_modelo = preparar_dataframe_para_modelo(entrada, feature_names)
    proba = float(model.predict_proba(df_modelo)[0][1])
    nivel = classificar_risco(proba)

    st.subheader("Resultado")
    st.metric("Probabilidade de risco", f"{proba:.1%}")
    st.metric("Nível", nivel)

# ---------------- PREVISÃO EM LOTE ----------------

st.divider()
st.header("📁 Análise de turma (upload de planilha)")

st.write("### 📥 Baixar modelo de preenchimento")
st.caption("Baixe a planilha modelo em .xlsx, preencha os dados e envie no campo abaixo.")

if hasattr(st, "link_button"):
    st.link_button("Baixar planilha modelo (.xlsx)", TEMPLATE_PLANILHA_URL)
else:
    st.markdown(f"[Baixar planilha modelo (.xlsx)]({TEMPLATE_PLANILHA_URL})")

arquivo = st.file_uploader("Enviar planilha (.xlsx)", type=["xlsx"])

if arquivo:
    try:
        df = pd.read_excel(arquivo)
    except Exception as exc:
        st.error("Não foi possível ler a planilha enviada.")
        st.exception(exc)
        st.stop()

    df = validar_e_normalizar_planilha(df)

    df_modelo = preparar_dataframe_para_modelo(df, feature_names)
    df["Probabilidade_Risco"] = model.predict_proba(df_modelo)[:, 1]
    df["Nivel_Risco"] = df["Probabilidade_Risco"].apply(classificar_risco)

    df = df.sort_values(by="Probabilidade_Risco", ascending=False)

    st.subheader("Alunos ordenados por risco")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="⬇️ Baixar resultado",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resultado_risco_alunos.csv",
        mime="text/csv",
    )
