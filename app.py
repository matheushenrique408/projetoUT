import os
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- configuração ---
DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

st.set_page_config(page_title="Mushroom IA - Form integrado", layout="wide")
st.title("Mushroom IA — Formulário integrado (previsão)")
st.markdown("Use o formulário abaixo para informar características do cogumelo. Treine o modelo (abaixo) antes de submeter o formulário para previsão.")

# --- carregamento robusto com diagnóstico e upload ---
def load_data(path=DATA_PATH):
    st.write("**Diagnóstico rápido**")
    st.write("Diretório atual:", os.getcwd())
    try:
        files = sorted(os.listdir("."))
    except Exception as e:
        files = ["(não foi possível listar arquivos: {})".format(e)]
    st.write("Arquivos neste diretório:", files)

    # 1) tenta ler do caminho configurado
    if os.path.exists(path):
        try:
            df_local = pd.read_csv(path)
            st.success(f"Arquivo encontrado: {path} (lido com sucesso).")
            return df_local
        except Exception as e:
            st.error(f"Arquivo '{path}' encontrado, mas ocorreu erro ao ler: {e}")
            st.info("Você pode tentar enviar o CSV via upload abaixo para teste.")

    # 2) se não existir ou falhar na leitura, permitir upload via UI
    st.warning(f"Arquivo '{path}' não encontrado no diretório atual (ou falha na leitura). Faça upload temporário do CSV para testar o app.")
    uploaded = st.file_uploader("Envie mushroom.csv (ou outro CSV compatível)", type=["csv"])
    if uploaded is not None:
        try:
            df_uploaded = pd.read_csv(uploaded)
            st.success("CSV carregado via upload com sucesso (apenas sessão atual).")
            return df_uploaded
        except Exception as e:
            st.error(f"Falha ao ler o CSV enviado: {e}")
            st.stop()

    # 3) fallback: instrução e parada
    st.error(f"Coloque '{path}' na pasta do projeto (ou faça upload). O app não pode prosseguir sem os dados.")
    st.stop()

# --- utilitários de pré-processamento ---
def is_boolean_like(series):
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
            mapping_vals = {
                True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
                't': 1, 'f': 0, 'y': 1, 'n': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0
            }
            X_proc[col] = ser.map(mapping_vals).fillna(0).astype(int)
        else:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(ser.astype(str))
            encoders[col] = le
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))
    return X_proc, y_enc, encoders, target_le

# --- execução principal ---
df = load_data()
st.subheader("Amostra dos dados (5 primeiras linhas)")
st.dataframe(df.head())

with st.expander("Visão geral das colunas e tipos"):
    info = pd.DataFrame({
        'coluna': list(df.columns),
        'tipo': [str(t) for t in df.dtypes],
        'valores_únicos': [df[c].nunique() for c in df.columns]
    })
    st.dataframe(info)

# preprocess para obter encoders e colunas de features
if TARGET_COL not in df.columns:
    st.error(f"A coluna alvo '{TARGET_COL}' não foi encontrada no CSV. Verifique o arquivo.")
    st.stop()

X_proc, y_enc, encoders, target_le = preprocess(df, TARGET_COL)
feature_columns = list(X_proc.columns)

st.subheader("Treinamento do modelo")
test_size = st.slider("Tamanho do conjunto de teste (%)", 5, 50, 20)
n_estimators = st.number_input("Número de árvores (n_estimators)", min_value=10, max_value=1000, value=100, step=10)
max_depth = st.number_input("Max depth (0 = None)", min_value=0, max_value=100, value=0, step=1)

if st.button("Treinar modelo agora"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y_enc, test_size=test_size/100.0, random_state=42, stratify=y_enc
        )
    except Exception as e:
        st.error(f"Erro ao dividir os dados: {e}")
        st.stop()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=(None if max_depth == 0 else int(max_depth)),
        random_state=42
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Treinado! Acurácia no teste: {acc:.4f}")
    st.text("Relatório de classificação:")
    st.text(classification_report(y_test, preds, target_names=target_le.classes_))
    st.session_state['model'] = rf
    st.session_state['encoders'] = encoders
    st.session_state['feature_columns'] = feature_columns
    st.session_state['target_le'] = target_le
else:
    if 'model' in st.session_state:
        st.info("Modelo encontrado na sessão. Pronto para previsões.")
    else:
        st.info("Treine o modelo para habilitar previsões com o formulário.")

st.markdown("---")
st.header("Formulário (preencha e envie para análise)")

# formulário com os mesmos campos; organiza respostas no dicionário 'q'
with st.form(key='mushroom_form'):
    q = {}
    q['class'] = st.radio(
        "1) Qual a classe do cogumelo? (class)",
        options=[("poisonous / venenoso", 'p'), ("edible / comestível", 'e')],
        format_func=lambda x: x[0]
    )

    q['cap-shape'] = st.radio(
        "2) Qual o formato do chapéu (cap-shape)?",
        options=[('Convexo / convex','x'),('Sino / bell','b'),('Afundado / sunken','s'),
                 ('Plano / flat','f'),('Protuberância / knobbed','k'),('Cônico / conical','c')],
        format_func=lambda x: x[0]
    )
    q['cap-surface'] = st.radio(
        "3) Superfície do chapéu (cap-surface)",
        options=[('Lisa / smooth','s'),('Escamosa / scaly','y'),('Fibrosa / fibrous','f'),('Sulcada / grooves','g')],
        format_func=lambda x: x[0]
    )
    q['cap-color'] = st.radio(
        "4) Cor do chapéu (cap-color)",
        options=[('Marrom / brown','n'),('Amarelo / yellow','y'),('Branco / white','w'),
                 ('Cinza / gray','g'),('Vermelho / red','e'),('Rosa / pink','p'),
                 ('Bege / buff','b'),('Roxo / purple','u'),('Canela / cinnamon','c'),
                 ('Verde / green','r')],
        format_func=lambda x: x[0]
    )
    q['bruises'] = st.radio("5) Escurece ao toque / bruises", options=[('Sim / yes','t'),('Não / no','f')], format_func=lambda x: x[0])
    q['odor'] = st.radio(
        "6) Odor",
        options=[('Cheiro forte / pungent','p'),('Amêndoas / almond','a'),('Anis / anise','l'),
                 ('Nenhum / none','n'),('Fétido / foul','f'),('Creosoto / creosote','c'),
                 ('Peixe / fishy','y'),('Apimentado / spicy','s'),('Mofo / musty','m')],
        format_func=lambda x: x[0]
    )
    q['gill-attachment'] = st.radio("7) Fixação das lâminas (gill-attachment)", options=[('Livre / free','f'),('Presa / attached','a')], format_func=lambda x: x[0])
    q['gill-spacing'] = st.radio("8) Espaçamento das lâminas (gill-spacing)", options=[('Próximas / close','c'),('Muito próximas / crowded','w')], format_func=lambda x: x[0])
    q['gill-size'] = st.radio("9) Tamanho das lâminas (gill-size)", options=[('Estreitas / narrow','n'),('Largas / broad','b')], format_func=lambda x: x[0])
    q['gill-color'] = st.radio(
        "10) Cor das lâminas (gill-color)",
        options=[('Preta / black','k'),('Marrom / brown','n'),('Cinza / gray','g'),('Rosa / pink','p'),
                 ('Branca / white','w'),('Chocolate','h'),('Roxa / purple','u'),('Vermelha / red','e'),
                 ('Bege / buff','b'),('Verde / green','r')],
        format_func=lambda x: x[0]
    )
    q['stalk-shape'] = st.radio("11) Formato do caule (stalk-shape)", options=[('Alargado na base / enlargi]()
