from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preparar_base_pede import carregar_base_unificada

MODEL_PATH = Path(__file__).with_name("modelo_risco.pkl")
COLUNA_INGLES = "Ingl\u00eas"
ROTULO_CRITICO = "Cr\u00edtico"

COLUNAS_ENTRADA = [
    "IDA",
    "IEG",
    "IPV",
    "Matem",
    "Portug",
    COLUNA_INGLES,
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
    COLUNA_INGLES: (0.0, 10.0),
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
        COLUNA_INGLES: 8.0,
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
        COLUNA_INGLES: 3.0,
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
        COLUNA_INGLES: 6.0,
        "IAA": 6.0,
        "IPS": 6.5,
        "IAN": 1.0,
        "Defas": 0.0,
    },
]

st.set_page_config(
    page_title="Dashboard Educacional - Passos Mágicos",
    page_icon="\U0001f4ca",
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


def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA

    texto = " ".join(str(valor).strip().split())
    return texto or pd.NA


def normalizar_pedra(valor):
    texto = normalizar_texto(valor)
    if pd.isna(texto):
        return pd.NA

    mapa = {
        "Agata": "\u00c1gata",
        "\u00c1gata": "\u00c1gata",
        "Ametista": "Ametista",
        "Quartzo": "Quartzo",
        "Top\u00e1zio": "Top\u00e1zio",
        "INCLUIR": pd.NA,
    }
    return mapa.get(texto, texto)


def normalizar_fase_ideal(valor):
    texto = normalizar_texto(valor)
    if pd.isna(texto):
        return pd.NA

    return texto.replace("°", "\u00ba")


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    df = carregar_base_unificada().copy()
    df["Ano"] = df["Ano Referencia"].astype(str)
    df["Pedra"] = df["Pedra Atual"].apply(normalizar_pedra)
    df["INDE"] = pd.to_numeric(df["INDE Atual"], errors="coerce")

    if "Fase ideal" in df.columns:
        df["Fase ideal"] = df["Fase ideal"].apply(normalizar_fase_ideal)
    if "Atingiu PV" in df.columns:
        df["Atingiu PV"] = df["Atingiu PV"].apply(normalizar_texto)

    colunas_numericas = [
        "IAA",
        "IEG",
        "IPS",
        "IDA",
        "IPV",
        "IAN",
        "Matem",
        "Portug",
        COLUNA_INGLES,
        "INDE",
        "Defas",
    ]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["RA", "Ano"], keep="first")
    df = df.replace(["", "NA", "-"], pd.NA)
    df = df[df["Pedra"].notna()].copy()

    indicadores = ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]
    for col in indicadores:
        if col in df.columns:
            df[col] = df[col].clip(0, 10)

    df["Nivel Defasagem"] = pd.cut(
        df["IAN"],
        bins=[0, 4, 7, 10],
        labels=["Alta defasagem", "Defasagem moderada", "Adequado"],
        include_lowest=True,
    )

    df["Score Educacional"] = (
        df["IEG"].fillna(0) * 0.2
        + df["IPS"].fillna(0) * 0.2
        + df["IDA"].fillna(0) * 0.3
        + df["IPV"].fillna(0) * 0.2
        + df["IAN"].fillna(0) * 0.1
    )

    return df


def classificar_risco(probabilidade: float) -> str:
    if probabilidade < 0.30:
        return "Baixo"
    if probabilidade < 0.60:
        return "Moderado"
    if probabilidade < 0.80:
        return "Alto"
    return ROTULO_CRITICO


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
            st.error(f"A coluna '{coluna}' deve conter apenas numeros.")
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

    st.title("Predição de Risco Educacional - Passos Mágicos")
    st.caption(
        "Sistema de alerta precoce para identificação de alunos em risco educacional."
    )

    st.header("Análise individual do aluno")

    with st.sidebar:
        st.header("Entrada manual")

        IDA = st.number_input("IDA", 0.0, 10.0, 6.0, 0.1)
        IEG = st.number_input("IEG", 0.0, 10.0, 6.0, 0.1)
        IPV = st.number_input("IPV", 0.0, 10.0, 6.0, 0.1)
        Matem = st.number_input("Matem", 0.0, 10.0, 6.0, 0.1)
        Portug = st.number_input("Portug", 0.0, 10.0, 6.0, 0.1)
        Ingles = st.number_input(COLUNA_INGLES, 0.0, 10.0, 6.0, 0.1)
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
        st.metric("Nível de risco", nivel)

    st.divider()
    st.header("Análise de turma")
    st.write("### Modelo de preenchimento")
    st.caption(
        "Baixe uma planilha modelo já preenchida com exemplos válidos. "
        "Você pode editar as linhas existentes ou substituí-las pelos seus alunos."
    )

    st.download_button(
        label="Baixar planilha modelo (.xlsx)",
        data=gerar_planilha_modelo(),
        file_name="template_predicao_risco.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.write("Colunas obrigatórias:")
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

        st.subheader("Alunos priorizados por risco")
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
        st.header("Filtros")
        ano_selecionado = st.selectbox(
            "Selecione o ano", ["Todos", "2022", "2023", "2024"]
        )

    if ano_selecionado != "Todos":
        df = df[df["Ano"] == ano_selecionado]

    st.title("📊 Dashboard Educacional - Passos Mágicos")
    st.write(
        """
Este dashboard apresenta uma análise dos indicadores educacionais dos alunos do programa **Passos Mágicos**.

Os gráficos exploram engajamento, desempenho acadêmico, fatores psicossociais e risco educacional.

Criamos um score educacional agregando múltiplos indicadores para sintetizar o desempenho global do aluno e utilizamos o dataset PEDE para todo o desenvolvimento deste dashboard.
"""
    )

    st.header("Visão Geral dos Indicadores")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de alunos", int(df["RA"].nunique()))
    col2.metric("Registros analisados", int(len(df)))
    col3.metric(
        "INDE médio", f"{df['INDE'].mean():.2f}" if df["INDE"].notna().any() else "-"
    )
    col4.metric("Engajamento médio", f"{df['IEG'].mean():.2f}")
    col5.metric("Aprendizagem média", f"{df['IDA'].mean():.2f}")

    if ano_selecionado == "Todos":
        st.header("Evolução dos indicadores ao longo dos anos")
        st.caption(
            """
    Este gráfico apresenta a evolução média dos principais indicadores educacionais ao longo dos anos analisados no programa.

    Os indicadores incluem engajamento dos alunos (IEG), fatores psicossociais (IPS), desempenho acadêmico (IDA), ponto de virada educacional (IPV) e adequação ao nível esperado (IAN).

    A análise da evolução ao longo do tempo permite identificar tendências de melhoria ou queda nesses indicadores, ajudando a avaliar o impacto do programa educacional no desenvolvimento dos alunos.
    """
        )
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
        st.caption(
            """
    Este gráfico apresenta o número de alunos únicos acompanhados em cada ano do programa Passos Mágicos.
    Ele permite visualizar o crescimento ou variação do número de estudantes atendidos ao longo do tempo.
    """
        )
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

    st.header("Distribuição das Pedras")
    st.caption(
        """
As **Pedras** representam o nível de desenvolvimento educacional do aluno dentro do programa.

Quartzo → maior defasagem educacional  
Ágata → nível intermediário de desenvolvimento  
Ametista → bom desempenho educacional  
Topázio → alunos destaque
"""
    )
    st.plotly_chart(px.histogram(df, x="Pedra", color="Pedra"), use_container_width=True)

    st.header("Adequação ao nível (IAN)")
    st.caption(
        """
O **IAN (Indicador de Adequação ao Nível)** mede se o aluno está no nível educacional esperado para sua idade ou série escolar.

Valores mais baixos indicam maior defasagem educacional, enquanto valores mais altos indicam que o estudante está mais próximo do nível adequado.

A distribuição desse indicador permite identificar o grau de adequação educacional dos alunos e avaliar a presença de possíveis lacunas de aprendizagem.
"""
    )
    st.plotly_chart(px.histogram(df, x="IAN", color="Pedra"), use_container_width=True)

    st.header("Classificação de Defasagem Educacional")
    st.caption(
        """
Esta visualização classifica os alunos de acordo com o nível de defasagem educacional com base no indicador de adequação ao nível (IAN).

Os estudantes são agrupados em três categorias principais:

• Alta defasagem educacional  
• Defasagem moderada  
• Nível educacional adequado

Essa classificação facilita a identificação de grupos de alunos que podem demandar maior atenção pedagógica ou estratégias de apoio educacional.
"""
    )
    st.plotly_chart(
        px.histogram(df, x="Nivel Defasagem", color="Nivel Defasagem"),
        use_container_width=True,
    )

    st.header("Engajamento x aprendizagem")
    st.caption(
        """
Este gráfico compara o nível de engajamento dos alunos nas atividades do programa com seu desempenho acadêmico.

O **IEG (Indicador de Engajamento)** representa o nível de participação dos alunos nas atividades educacionais, enquanto o **IDA (Indicador de Desempenho Acadêmico)** mede o resultado educacional obtido.

A análise conjunta desses indicadores permite investigar se alunos mais engajados tendem a apresentar melhor desempenho acadêmico.
"""
    )
    st.plotly_chart(
        px.scatter(df, x="IEG", y="IDA", color="Pedra", opacity=0.6),
        use_container_width=True,
    )

    st.header("Autoavaliação vs Desempenho")
    st.caption(
        """
O **IAA (Indicador de Autoavaliação)** representa como os próprios alunos percebem seu desempenho educacional.

Ao comparar esse indicador com o **IDA (Indicador de Desempenho Acadêmico)**, é possível avaliar se a percepção dos estudantes sobre seu aprendizado está alinhada com seu desempenho real.

A linha de referência no gráfico representa o ponto em que percepção e desempenho são equivalentes.
"""
    )
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

    st.header("Indicador Psicossocial")
    st.caption(
        """
O **IPS (Indicador Psicossocial)** mede fatores emocionais, sociais e comportamentais que podem influenciar o processo de aprendizagem dos alunos.

Esses fatores incluem aspectos como motivação, bem-estar emocional e ambiente social.

A análise desse indicador ajuda a compreender como fatores não diretamente acadêmicos podem impactar o desempenho educacional.
"""
    )
    st.plotly_chart(px.box(df, x="Pedra", y="IPS", color="Pedra"), use_container_width=True)

    st.header("Ponto de Virada")
    st.caption(
        """
O **IPV (Indicador de Ponto de Virada)** identifica momentos em que o aluno apresenta mudanças relevantes em seu comportamento ou desempenho educacional.

Esses pontos podem indicar fases de evolução significativa, melhoria no engajamento ou mudanças positivas no processo de aprendizagem.

A comparação entre alunos que atingiram ou não esse ponto permite analisar possíveis impactos no desempenho acadêmico.
"""
    )
    st.plotly_chart(
        px.box(df, x="Atingiu PV", y="IDA", color="Atingiu PV"),
        use_container_width=True,
    )

    st.header("Notas Escolares")
    st.caption(
        """
Esta análise apresenta a distribuição das notas dos alunos nas principais disciplinas escolares.

A comparação entre as disciplinas permite identificar áreas em que os estudantes apresentam maior desempenho ou maior dificuldade.

Essa informação pode auxiliar na identificação de possíveis lacunas de aprendizagem em áreas específicas do conhecimento.
"""
    )
    df_notas = df.melt(
        value_vars=["Matem", "Portug", COLUNA_INGLES],
        var_name="Disciplina",
        value_name="Nota",
    )
    st.plotly_chart(
        px.box(df_notas, x="Disciplina", y="Nota", color="Disciplina"),
        use_container_width=True,
    )

    st.header("Score Educacional Geral")
    st.caption(
        """
O **Score Educacional** foi criado para sintetizar o desempenho global dos alunos a partir da combinação de diferentes indicadores educacionais.

Esse índice agrega informações sobre engajamento, aspectos psicossociais, desempenho acadêmico, ponto de virada educacional e adequação ao nível.

A análise desse score permite observar de forma mais integrada o desenvolvimento educacional dos estudantes.
"""
    )
    st.plotly_chart(
        px.histogram(df, x="Score Educacional", color="Pedra"),
        use_container_width=True,
    )

    st.header("Correlação entre Indicadores")
    st.caption(
        """
A matriz de correlação apresenta o grau de relação entre os principais indicadores educacionais analisados.

Valores próximos de **1** indicam forte relação positiva entre os indicadores, enquanto valores próximos de **0** indicam baixa relação.

Essa análise ajuda a compreender como diferentes fatores educacionais se influenciam mutuamente.
"""
    )
    corr = df[["IEG", "IPS", "IDA", "IPV", "IAN", "IAA"]].corr(numeric_only=True)
    st.plotly_chart(
        px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", aspect="auto"),
        use_container_width=True,
    )

    st.header("Previsão de Risco Educacional")
    st.caption(
        """ Foi desenvolvido um modelo de **Machine Learning (Random Forest)** para prever o risco educacional dos alunos. 

O modelo utiliza indicadores como **engajamento (IEG), desempenho acadêmico (IDA), fatores psicossociais (IPS) e ponto de virada (IPV)**. 

Com base nesses indicadores, o modelo identifica padrões associados ao risco educacional e permite antecipar alunos que podem apresentar dificuldades de aprendizagem, auxiliando na priorização de intervenções pedagógicas. """
    )
    st.caption(
        """ O modelo apresentou alta acurácia na identificação do risco educacional. 
Entretanto, é importante considerar que os resultados dependem da definição da variável de risco e da distribuição dos dados no conjunto analisado. """
    )

    df_ml = df.dropna(subset=["IAN", "IEG", "IPS", "IDA", "IPV"]).copy()
    if not df_ml.empty and df_ml["RA"].nunique() > 1:
        df_ml["risco"] = ((df_ml["IAN"] < 5) | (df_ml["IDA"] < 5)).astype(int)
        features = ["IEG", "IPS", "IDA", "IPV"]

        X = df_ml[features]
        y = df_ml["risco"]

        if y.nunique() > 1 and len(df_ml) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            st.metric("Acurácia do modelo", f"{acc:.2%}")
            st.text(classification_report(y_test, pred))

            importance = pd.DataFrame(
                {
                    "Variável": features,
                    "Importância": model.feature_importances_,
                }
            ).sort_values("Importância", ascending=False)

            fig = px.bar(
                importance,
                x="Variável",
                y="Importância",
                color="Variável",
                text="Importância",
            )

            st.caption(
                """ O gráfico a seguir mostra a **importância dos indicadores utilizados pelo modelo de Machine Learning para prever o risco educacional**. 

Cada barra representa o quanto um indicador contribui para a capacidade do modelo de identificar alunos em risco.
Indicadores com maior importância têm maior impacto na previsão e podem ser considerados fatores críticos para monitoramento e intervenção educacional. """
            )

            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Não há dados suficientes com duas classes de risco para treinar o modelo no recorte selecionado."
            )
    else:
        st.info("Não há dados suficientes para calcular a previsão de risco neste recorte.")

    st.header("Insights")
    st.caption(
        """
Esta seção apresenta algumas relações observadas entre os indicadores educacionais analisados.

A análise das correlações entre engajamento, fatores psicossociais e desempenho acadêmico ajuda a compreender melhor quais fatores podem estar mais associados ao sucesso educacional dos alunos.
"""
    )

    corr1 = df[["IEG", "IDA"]].corr().iloc[0, 1]
    corr2 = df[["IPS", "IDA"]].corr().iloc[0, 1]

    st.write("Correlação Engajamento x Desempenho:", round(corr1, 2))
    st.write("Correlação Psicossocial x Desempenho:", round(corr2, 2))


with st.sidebar:
    st.header("Navegação")
    pagina = st.radio("Escolha a página", ["Predição de risco", "Dashboard analítico"])

if pagina == "Predição de risco":
    render_predicao_page()
else:
    render_dashboard_page()
