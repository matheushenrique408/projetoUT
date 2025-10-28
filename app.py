
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

st.set_page_config(page_title="Mushroom IA - Classificação", layout="wide")

st.title("Mushroom IA — Previsão: Comestível ou Venenoso")
st.markdown("App que carrega um dataset de cogumelos (`mushroom.csv`), treina um modelo dentro do app e permite prever se um cogumelo é comestível (e) ou venenoso (p).")

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo '{path}' não encontrado. Coloque o CSV na pasta do projeto.")
    df = pd.read_csv(path)
    return df

def is_boolean_like(series):
    # detecta séries booleanas mesmo como strings 'True'/'False' ou '0'/'1'
    unique = set(series.dropna().unique())
    bool_like_sets = [
        {True, False},
        {"True", "False"},
        {"true", "false"},
        {0, 1},
        {"0", "1"},
        {"t", "f"},
        {"y", "n"},
        {"yes", "no"}
    ]
    for s in bool_like_sets:
        if unique.issubset(s):
            return True
    return False

def preprocess(df, target_col=TARGET_COL):
    encoders = {}
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    X_proc = pd.DataFrame(index=X.index)

    for col in X.columns:
        ser = X[col]
        if is_boolean_like(ser):
            # map common boolean-like values to 0/1
            mapping = {}
            unique = list(ser.dropna().unique())
            # try to coerce numeric
            try:
                numeric = pd.to_numeric(ser.dropna())
                if set(numeric.unique()).issubset({0,1}):
                    X_proc[col] = ser.astype(int)
                    continue
            except Exception:
                pass
            # fallback mapping for strings
            mapping_vals = {
                True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0,
                't':1,'f':0,'y':1,'n':0,'yes':1,'no':0,'1':1,'0':0
            }
            X_proc[col] = ser.map(mapping_vals).fillna(0).astype(int)
        else:
            # treat as categorical: label encode
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(ser.astype(str))
            encoders[col] = le

    # encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))

    return X_proc, y_enc, encoders, target_le

def build_sidebar_inputs(df, encoders):
    st.sidebar.header("Características do cogumelo para prever")
    inputs = {}
    X = df.drop(columns=[TARGET_COL])
    for col in X.columns:
        ser = X[col]
        if is_boolean_like(ser):
            # decide default based on most common
            most_common = ser.mode().iloc[0] if not ser.mode().empty else None
            default = False
            if isinstance(most_common, (int, float)):
                default = bool(most_common)
            else:
                default = str(most_common).lower() in ['true','t','y','yes','1']
            val = st.sidebar.checkbox(f"{col} (boolean)", value=default)
            inputs[col] = int(val)
        else:
            # categorical: use selectbox with unique options from data
            uniques = list(ser.dropna().unique())
            # convert to strings for display
            uniques_str = [str(u) for u in uniques]
            default = uniques_str[0] if uniques_str else ""
            choice = st.sidebar.selectbox(f"{col}", options=uniques_str, index=0)
            # transform choice using encoder if exists
            if col in encoders:
                enc = encoders[col]
                try:
                    transformed = int(enc.transform([str(choice)])[0])
                except Exception:
                    # unseen value -> add fallback mapping (map to 0)
                    transformed = 0
            else:
                transformed = 0
            inputs[col] = transformed
    return inputs

# Load data
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.subheader("Amostra dos dados (5 primeiras linhas)")
st.dataframe(df.head())

with st.expander("Visão geral das colunas e tipos"):
    info = pd.DataFrame({'coluna': df.columns, 'tipo': [str(t) for t in df.dtypes], 'valores_únicos': [df[c].nunique() for c in df.columns]})
    st.dataframe(info)

# Basic plot: distribuição da classe alvo
if TARGET_COL in df.columns:
    fig = px.histogram(df, x=TARGET_COL, title="Distribuição da variável alvo (class)")
    st.plotly_chart(fig, use_container_width=True)

# Preprocess and train
st.subheader("Treinamento do modelo")
st.markdown("O app irá pré-processar automaticamente colunas char/bool e treinar um RandomForestClassifier.")
X_proc, y_enc, encoders, target_le = preprocess(df, TARGET_COL)

test_size = st.slider("Tamanho do conjunto de teste (%)", 5, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=test_size/100.0, random_state=42, stratify=y_enc)

n_estimators = st.number_input("Número de árvores (n_estimators)", min_value=10, max_value=1000, value=100, step=10)
max_depth = st.number_input("Max depth (0 = None)", min_value=0, max_value=100, value=0, step=1)

if st.button("Treinar modelo agora"):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth==0 else int(max_depth)), random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Treinado! Acurácia no teste: {acc:.4f}")
    st.text("Relatório de classificação:")
    st.text(classification_report(y_test, preds, target_names=target_le.classes_))
    # store model in session state for reuse
    st.session_state['model'] = rf
else:
    # try to show a placeholder if already trained in session
    if 'model' in st.session_state:
        st.info("Modelo carregado da sessão anterior.")
    else:
        st.info("Clique em 'Treinar modelo agora' para treinar com os hiperparâmetros acima.")

# Sidebar inputs for prediction
if 'model' in st.session_state:
    model = st.session_state['model']
    user_inputs = build_sidebar_inputs(df, encoders)
    # build feature vector in the same column order as X_proc
    feature_vec = [user_inputs[c] if c in user_inputs else 0 for c in X_proc.columns]
    feature_arr = np.array(feature_vec).reshape(1, -1)
    pred = model.predict(feature_arr)[0]
    proba = model.predict_proba(feature_arr)[0] if hasattr(model, "predict_proba") else None
    pred_label = target_le.inverse_transform([pred])[0]
    st.subheader("Resultado da previsão")
    if pred_label.lower().startswith('e'):
        st.success(f"🍽️ Previsto: COMESTÍVEL (label = {pred_label})")
    else:
        st.error(f"☠️ Previsto: VENENOSO (label = {pred_label})")
    if proba is not None:
        # show probability for each class
        prob_df = pd.DataFrame({'classe': target_le.classes_, 'probabilidade': proba})
        st.table(prob_df)
else:
    st.info("Treine o modelo primeiro para habilitar previsões.")

st.markdown("""---
**Notas:**  
- O app detecta automaticamente colunas boolean-like e as converte para 0/1.  
- Colunas char/categóricas são codificadas com LabelEncoder.  
- Se quiser que eu salve o modelo em disco (`.joblib`), posso adicionar essa funcionalidade.
""")
