from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from preparar_base_pede import carregar_base_unificada

MODEL_PATH = Path(__file__).with_name("modelo_risco.pkl")

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

EXEMPLOS_TEMPLATE = [
    {
        "IDA": 8.0,
        "IEG": 8.0,
        "IPV": 8.0,
        "Matem": 8.0,
        "Portug": 8.0,
        "Inglês": 8.0,
        "IAA": 8.0,
        "IPS": 8.0,
        "IAN": 5.0,
        "Defas": 0.0,
    },
    {
        "IDA": 3.0,
        "IEG": 3.0,
        "IPV": 3.0,
        "Matem": 3.0,
        "Portug": 3.0,
        "Inglês": 3.0,
        "IAA": 3.0,
        "IPS": 3.0,
        "IAN": -2.0,
        "Defas": 2.0,
    },
    {
        "IDA": 6.0,
        "IEG": 5.5,
        "IPV": 6.5,
        "Matem": 6.0,
        "Portug": 5.5,
        "Inglês": 6.0,
        "IAA": 6.0,
        "IPS": 6.5,
        "IAN": 1.0,
        "Defas": 0.0,
    },
]

st.set_page_config(
    page_title="Risco Educacional - Passos Magicos",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def gerar_planilha_modelo() -> bytes:
    df_template = pd.DataFrame(EXEMPLOS_TEMPLATE, columns=COLUNAS_ENTRADA)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_template.to_excel(writer, index=False, sheet_name="modelo")
    return buffer.getvalue()


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    df = carregar_base_unificada().copy()
    df["Ano"] = df["Ano Referencia"].astype(str)
    df["Pedra"] = df["Pedra Atual"]
    df["INDE"] = pd.to_numeric(df["INDE Atual"], errors="coerce")

    colunas_numericas = [
        "IAA",
        "IEG",
        "IPS",
        "IDA",
        "IPV",
        "IAN",
        "Matem",
        "Portug",
        "Inglês",
        "INDE",
        "Defas",
    ]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["RA", "Ano"], keep="first")
    df = df.replace(["", "NA", "-"], pd.NA)

    df["Nivel Defasagem"] = pd.cut(
        df["Defas"],
        bins=[-10, -1, 1, 10],
        labels=["Adiantado", "Adequado", "Defasado"],
        include_lowest=True,
    )

    df["Score Educacional"] = (
        df["IEG"].fillna(0) * 0.2
        + df["IPS"].fillna(0) * 0.15
        + df["IDA"].fillna(0) * 0.3
        + df["IPV"].fillna(0) * 0.2
        + ((10 - df["Defas"].clip(-10, 10).abs()).fillna(0)) * 0.15
    )

    return df


def classificar_risco(probabilidade: float) -> str:
    if probabilidade < 0.30:
        return "Baixo"
    if probabilidade < 0.60:
        return "Moderado"
    if probabilidade < 0.80:
        return "Alto"
    return "Critico"


def preparar_dataframe_para_modelo(
    df_usuario: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    df_modelo = pd.DataFrame(0, index=df_usuario.index, columns=feature_names)

    for col in df_usuario.columns:
        if col in df_modelo.columns:
            df_modelo[col] = df_usuario[col]

    return df_modelo.fillna(0)


def validar_e_normalizar_planilha(df: pd.DataFrame) -> pd.DataFrame:
    faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    if faltando:
        st.error("A planilha esta invalida.")
        st.write("Colunas obrigatorias faltando:")
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
            st.error(f"A coluna '{coluna}' deve conter apenas numeros.")
            st.write(
                "Linhas com valores invalidos (1-based, ate 10):",
                [i + 2 for i in linhas_nao_numericas[:10]],
            )
            st.stop()

        linhas_vazias = df_validado.index[serie_numerica.isna()].tolist()
        if linhas_vazias:
            st.error(f"A coluna '{coluna}' possui valores vazios.")
            st.write(
                "Linhas com valores ausentes (1-based, ate 10):",
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
                "Linhas fora do intervalo (1-based, ate 10):",
                [i + 2 for i in fora_intervalo[:10]],
            )
            st.stop()

        df_validado[coluna] = serie_numerica.astype(float)

    return df_validado


def render_predicao_page() -> None:
    try:
        model = load_model()
    except Exception as exc:
        st.error("Nao foi possivel carregar o modelo de risco.")
        st.code(f"Arquivo esperado: {MODEL_PATH}")
        st.exception(exc)
        st.stop()

    if not hasattr(model, "feature_names_in_"):
        st.error("O modelo carregado nao possui 'feature_names_in_'.")
        st.stop()

    if not hasattr(model, "predict_proba"):
        st.error("O modelo carregado nao suporta 'predict_proba'.")
        st.stop()

    feature_names = list(model.feature_names_in_)

    st.title("Predicao de Risco Educacional - Passos Magicos")
    st.caption(
        "Sistema de alerta precoce para identificacao de alunos em risco educacional."
    )

    st.header("Analise individual do aluno")

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
        analisar = st.button("Calcular risco")

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
        st.metric("Nivel", nivel)

    st.divider()
    st.header("Analise de turma (upload de planilha)")
    st.write("### Baixar modelo de preenchimento")
    st.caption(
        "Baixe uma planilha modelo ja preenchida com exemplos validos. "
        "Voce pode editar as linhas existentes ou substitui-las pelos seus alunos."
    )

    st.download_button(
        label="Baixar planilha modelo (.xlsx)",
        data=gerar_planilha_modelo(),
        file_name="template_predicao_risco.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.write("Colunas obrigatorias:")
    st.code(", ".join(COLUNAS_ENTRADA))

    arquivo = st.file_uploader("Enviar planilha (.xlsx)", type=["xlsx"])

    if arquivo:
        try:
            df = pd.read_excel(arquivo)
        except Exception as exc:
            st.error("Nao foi possivel ler a planilha enviada.")
            st.exception(exc)
            st.stop()

        df = validar_e_normalizar_planilha(df)

        df_modelo = preparar_dataframe_para_modelo(df, feature_names)
        df["Probabilidade_Risco"] = model.predict_proba(df_modelo)[:, 1]
        df["Nivel_Risco"] = df["Probabilidade_Risco"].apply(classificar_risco)

        df = df.sort_values(by="Probabilidade_Risco", ascending=False)
        df_exibicao = df.copy()
        df_exibicao["Probabilidade_Risco"] = df_exibicao["Probabilidade_Risco"].map(
            lambda x: f"{x:.1%}"
        )

        st.subheader("Alunos ordenados por risco")
        st.dataframe(df_exibicao, use_container_width=True)

        st.download_button(
            label="Baixar resultado",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="resultado_risco_alunos.csv",
            mime="text/csv",
        )


def render_dashboard_page() -> None:
    df = load_dashboard_data()

    with st.sidebar:
        st.header("Filtros do dashboard")
        ano_selecionado = st.selectbox(
            "Selecione o ano", ["Todos", "2022", "2023", "2024"]
        )

    if ano_selecionado != "Todos":
        df = df[df["Ano"] == ano_selecionado]

    st.title("Dashboard Educacional - Passos Magicos")
    st.caption(
        "Painel analitico com os indicadores educacionais da base historica "
        "consolidada de 2022 a 2024."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de alunos", int(df["RA"].nunique()))
    col2.metric("Registros analisados", int(len(df)))
    col3.metric(
        "INDE medio", f"{df['INDE'].mean():.2f}" if df["INDE"].notna().any() else "-"
    )
    col4.metric("Engajamento medio", f"{df['IEG'].mean():.2f}")
    col5.metric("Aprendizagem media", f"{df['IDA'].mean():.2f}")

    if ano_selecionado == "Todos":
        st.header("Evolucao dos indicadores ao longo dos anos")
        evolucao = (
            df.groupby("Ano", as_index=False)[["IEG", "IPS", "IDA", "IPV", "IAN"]]
            .mean()
            .sort_values("Ano")
        )
        fig = px.line(
            evolucao,
            x="Ano",
            y=["IEG", "IPS", "IDA", "IPV", "IAN"],
            markers=True,
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

        st.header("Quantidade de alunos acompanhados por ano")
        alunos = (
            df.groupby("Ano", as_index=False)["RA"]
            .nunique()
            .sort_values("Ano")
            .rename(columns={"RA": "Alunos"})
        )
        fig = px.bar(alunos, x="Ano", y="Alunos", text="Alunos")
        fig.update_traces(textposition="outside")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Distribuicao das pedras")
    st.plotly_chart(px.histogram(df, x="Pedra", color="Pedra"), use_container_width=True)

    st.header("Adequacao ao nivel (IAN)")
    st.plotly_chart(px.histogram(df, x="IAN", color="Pedra"), use_container_width=True)

    st.header("Classificacao de defasagem educacional")
    st.plotly_chart(
        px.histogram(df, x="Nivel Defasagem", color="Nivel Defasagem"),
        use_container_width=True,
    )

    st.header("Engajamento vs aprendizagem")
    st.plotly_chart(
        px.scatter(df, x="IEG", y="IDA", color="Pedra", opacity=0.6),
        use_container_width=True,
    )

    st.header("Autoavaliacao vs desempenho")
    fig = px.scatter(df, x="IAA", y="IDA", color="Pedra", opacity=0.6)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=10,
        y1=10,
        line=dict(color="white", dash="dash"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("Indicador psicossocial")
    st.plotly_chart(px.box(df, x="Pedra", y="IPS", color="Pedra"), use_container_width=True)

    st.header("Ponto de virada")
    st.plotly_chart(
        px.box(df, x="Atingiu PV", y="IDA", color="Atingiu PV"),
        use_container_width=True,
    )

    st.header("Notas escolares")
    df_notas = df.melt(
        value_vars=["Matem", "Portug", "Inglês"],
        var_name="Disciplina",
        value_name="Nota",
    )
    st.plotly_chart(
        px.box(df_notas, x="Disciplina", y="Nota", color="Disciplina"),
        use_container_width=True,
    )

    st.header("Score educacional geral")
    st.plotly_chart(
        px.histogram(df, x="Score Educacional", color="Pedra"),
        use_container_width=True,
    )

    st.header("Correlacao entre indicadores")
    corr = df[["IEG", "IPS", "IDA", "IPV", "IAN", "IAA"]].corr(numeric_only=True)
    st.plotly_chart(
        px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", aspect="auto"),
        use_container_width=True,
    )


with st.sidebar:
    st.header("Navegacao")
    pagina = st.radio("Escolha a pagina", ["Predicao de risco", "Dashboard analitico"])

if pagina == "Predicao de risco":
    render_predicao_page()
else:
    render_dashboard_page()
