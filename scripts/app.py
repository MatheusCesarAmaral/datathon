import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risco Educacional - Passos Mágicos", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("modelo_risco.pkl")

model = load_model()

# ---------------- UTILIDADES ----------------

def classificar_risco(p: float) -> str:
    if p < 0.30:
        return "Baixo"
    elif p < 0.60:
        return "Moderado"
    elif p < 0.80:
        return "Alto"
    return "Crítico"

def preparar_dataframe_para_modelo(df_usuario, feature_names):
    """Garante que o dataset enviado tenha todas as colunas do modelo"""
    df_modelo = pd.DataFrame(columns=feature_names)
    df_modelo.loc[:, :] = 0

    for col in df_usuario.columns:
        if col in df_modelo.columns:
            df_modelo[col] = df_usuario[col]

    return df_modelo.fillna(0)

# ---------------- TÍTULO ----------------

st.title("📊 Predição de Risco Educacional — Passos Mágicos")
st.caption("Sistema de alerta precoce para identificação de alunos em risco educacional.")

# ---------------- PREVISÃO INDIVIDUAL ----------------

st.header("Análise individual do aluno")

feature_names = model.feature_names_in_

with st.sidebar:
    st.header("Entrada manual")

    IDA = st.number_input("IDA",0.0,10.0,6.0,0.1)
    IEG = st.number_input("IEG",0.0,10.0,6.0,0.1)
    IPV = st.number_input("IPV",0.0,10.0,6.0,0.1)
    Matem = st.number_input("Matem",0.0,10.0,6.0,0.1)
    Portug = st.number_input("Portug",0.0,10.0,6.0,0.1)
    Ingles = st.number_input("Inglês",0.0,10.0,6.0,0.1)
    IAA = st.number_input("IAA",0.0,10.0,6.0,0.1)
    IPS = st.number_input("IPS",0.0,10.0,6.0,0.1)
    IAN = st.number_input("IAN",-10.0,10.0,0.0,1.0)
    Defas = st.number_input("Defas",-10.0,10.0,0.0,1.0)

    st.divider()
    analisar = st.button("🔮 Calcular risco")

if analisar:

    entrada = pd.DataFrame([[IDA,IEG,IPV,Matem,Portug,Ingles,IAA,IPS,IAN,Defas]],
                           columns=["IDA","IEG","IPV","Matem","Portug","Inglês","IAA","IPS","IAN","Defas"])

    df_modelo = preparar_dataframe_para_modelo(entrada, feature_names)

    proba = model.predict_proba(df_modelo)[0][1]
    nivel = classificar_risco(proba)

    st.subheader("Resultado")
    st.metric("Probabilidade de risco", f"{proba:.1%}")
    st.metric("Nível", nivel)

# ---------------- PREVISÃO EM LOTE ----------------

st.divider()
st.header("📁 Análise de turma (upload de planilha)")

st.write("### 📥 Baixar modelo de preenchimento")

template = pd.DataFrame(columns=[
    "IDA","IEG","IPV","Matem","Portug","Inglês",
    "IAA","IPS","IAN","Defas"
])

st.download_button(
    label="Baixar planilha modelo",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="modelo_preenchimento_alunos.csv",
    mime="text/csv"
)

st.write("Preencha essa planilha e envie abaixo:")

arquivo = st.file_uploader("Enviar planilha (.xlsx)", type=["xlsx"])

if arquivo:

    df = pd.read_excel(arquivo)

    # ---------- VALIDAÇÃO ----------
    colunas_necessarias = [
        "IDA","IEG","IPV","Matem","Portug","Inglês",
        "IAA","IPS","IAN","Defas"
    ]

    faltando = [c for c in colunas_necessarias if c not in df.columns]

    if len(faltando) > 0:
        st.error("A planilha está inválida.")
        st.write("Colunas obrigatórias faltando:")
        st.write(faltando)
        st.stop()

    # valida valores numéricos
    for c in colunas_necessarias:
        if not pd.api.types.is_numeric_dtype(df[c]):
            st.error(f"A coluna '{c}' deve conter apenas números.")
            st.stop()

    # ---------- PREDIÇÃO ----------
    df_modelo = preparar_dataframe_para_modelo(df, feature_names)

    df["Probabilidade_Risco"] = model.predict_proba(df_modelo)[:,1]
    df["Nivel_Risco"] = df["Probabilidade_Risco"].apply(classificar_risco)

    df = df.sort_values(by="Probabilidade_Risco", ascending=False)

    st.subheader("Alunos ordenados por risco")
    st.dataframe(df)

    st.download_button(
        label="⬇️ Baixar resultado",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="resultado_risco_alunos.csv",
        mime="text/csv"
    )