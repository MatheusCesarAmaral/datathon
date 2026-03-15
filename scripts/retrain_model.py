from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preparar_base_pede import carregar_base_unificada

FEATURES_MODELO = [
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

MODEL_PATH = Path(__file__).with_name("modelo_risco.pkl")


def preparar_dados() -> tuple[pd.DataFrame, pd.Series]:
    df = carregar_base_unificada()
    df["risco"] = df["Pedra Atual"].isin(["Quartzo", "Ágata"]).astype(int)

    df_model = df[FEATURES_MODELO + ["risco"]].copy()
    for coluna in FEATURES_MODELO:
        df_model[coluna] = pd.to_numeric(df_model[coluna], errors="coerce")

    df_model = df_model.dropna(subset=FEATURES_MODELO + ["risco"])

    X = df_model[FEATURES_MODELO]
    y = df_model["risco"]
    return X, y


def treinar_modelo() -> None:
    X, y = preparar_dados()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("classification_report:")
    print(classification_report(y_test, pred))
    print("features:", list(model.feature_names_in_))

    joblib.dump(model, MODEL_PATH)
    print(f"modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    treinar_modelo()
