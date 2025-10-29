import os
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

st.set_page_config(page_title="Mushroom IA - Form integrado", layout="wide")
st.title("Mushroom IA — Formulário integrado (previsão)")
st.markdown("Use o formulário abaixo para informar características do cogumelo. Treine o modelo (abaixo) antes de submeter o formulário para previsão.")

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f\"Arquivo '{path}' não encontrado. Coloque o CSV na pasta do projeto.\")
        st.stop()
    return pd.read_csv(path)

def is_boolean_like(series):
    unique = set(series.dropna().unique())
    bool_like_sets = [
        {True, False},
        {\"True\", \"False\"},
        {\"true\", \"false\"},
        {0, 1},
        {\"0\", \"1\"},
        {\"t\", \"f\"},
        {\"y\", \"n\"},
        {\"yes\", \"no\"}
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
                True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0,
                't':1,'f':0,'y':1,'n':0,'yes':1,'no':0,'1':1,'0':0
            }
            X_proc[col] = ser.map(mapping_vals).fillna(0).astype(int)
        else:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(ser.astype(str))
            encoders[col] = le
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))
    return X_proc, y_enc, encoders, target_le

# --- load data ---
df = load_data()
st.subheader(\"Amostra dos dados (5 primeiras linhas)\")
st.dataframe(df.head())

with st.expander(\"Visão geral das colunas e tipos\"):
    info = pd.DataFrame({'coluna': df.columns, 'tipo': [str(t) for t in df.dtypes], 'valores_únicos': [df[c].nunique() for c in df.columns]})
    st.dataframe(info)

# preprocess once to get encoders and feature columns
X_proc, y_enc, encoders, target_le = preprocess(df, TARGET_COL)
feature_columns = list(X_proc.columns)

st.subheader(\"Treinamento do modelo\")
test_size = st.slider(\"Tamanho do conjunto de teste (%)\", 5, 50, 20)
n_estimators = st.number_input(\"Número de árvores (n_estimators)\", min_value=10, max_value=1000, value=100, step=10)
max_depth = st.number_input(\"Max depth (0 = None)\", min_value=0, max_value=100, value=0, step=1)

if st.button(\"Treinar modelo agora\"):
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=test_size/100.0, random_state=42, stratify=y_enc)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth==0 else int(max_depth)), random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f\"Treinado! Acurácia no teste: {acc:.4f}\")
    st.text(\"Relatório de classificação:\")
    st.text(classification_report(y_test, preds, target_names=target_le.classes_))
    st.session_state['model'] = rf
    st.session_state['encoders'] = encoders
    st.session_state['feature_columns'] = feature_columns
    st.session_state['target_le'] = target_le
else:
    if 'model' in st.session_state:
        st.info(\"Modelo encontrado na sessão. Pronto para previsões.\")
    else:
        st.info(\"Treine o modelo para habilitar previsões com o formulário.\")

st.markdown(\"---\")
st.header(\"Formulário (preencha e envie para análise)\")
# We'll present the same questions as radios. The first ('class') is included visually but not used for prediction.
with st.form(key='mushroom_form'):
    # question mapping: dataset_column -> (label, options list of (display_label, code))
    q = {}
    # 'class' - shown but not used
    q['class'] = st.radio(\"1) Qual a classe do cogumelo? (class)\", options=[(\"poisonous / venenoso\", 'p'), (\"edible / comestível\", 'e')], format_func=lambda x: x[0])

    q['cap-shape'] = st.radio(\"2) Qual o formato do chapéu (cap-shape)?\", options=[('Convexo / convex','x'),('Sino / bell','b'),('Afundado / sunken','s'),('Plano / flat','f'),('Protuberância / knobbed','k'),('Cônico / conical','c')], format_func=lambda x: x[0])
    q['cap-surface'] = st.radio(\"3) Superfície do chapéu (cap-surface)\", options=[('Lisa / smooth','s'),('Escamosa / scaly','y'),('Fibrosa / fibrous','f'),('Sulcada / grooves','g')], format_func=lambda x: x[0])
    q['cap-color'] = st.radio(\"4) Cor do chapéu (cap-color)\", options=[('Marrom / brown','n'),('Amarelo / yellow','y'),('Branco / white','w'),('Cinza / gray','g'),('Vermelho / red','e'),('Rosa / pink','p'),('Bege / buff','b'),('Roxo / purple','u'),('Canela / cinnamon','c'),('Verde / green','r')], format_func=lambda x: x[0])
    q['bruises'] = st.radio(\"5) Escurece ao toque / bruises\", options=[('Sim / yes','t'),('Não / no','f')], format_func=lambda x: x[0])
    q['odor'] = st.radio(\"6) Odor\", options=[('Cheiro forte / pungent','p'),('Amêndoas / almond','a'),('Anis / anise','l'),('Nenhum / none','n'),('Fétido / foul','f'),('Creosoto / creosote','c'),('Peixe / fishy','y'),('Apimentado / spicy','s'),('Mofo / musty','m')], format_func=lambda x: x[0])
    q['gill-attachment'] = st.radio(\"7) Fixação das lâminas (gill-attachment)\", options=[('Livre / free','f'),('Presa / attached','a')], format_func=lambda x: x[0])
    q['gill-spacing'] = st.radio(\"8) Espaçamento das lâminas (gill-spacing)\", options=[('Próximas / close','c'),('Muito próximas / crowded','w')], format_func=lambda x: x[0])
    q['gill-size'] = st.radio(\"9) Tamanho das lâminas (gill-size)\", options=[('Estreitas / narrow','n'),('Largas / broad','b')], format_func=lambda x: x[0])
    q['gill-color'] = st.radio(\"10) Cor das lâminas (gill-color)\", options=[('Preta / black','k'),('Marrom / brown','n'),('Cinza / gray','g'),('Rosa / pink','p'),('Branca / white','w'),('Chocolate','h'),('Roxa / purple','u'),('Vermelha / red','e'),('Bege / buff','b'),('Verde / green','r')], format_func=lambda x: x[0])
    q['stalk-shape'] = st.radio(\"11) Formato do caule (stalk-shape)\", options=[('Alargado na base / enlarging','e'),('Afunilando / tapering','t')], format_func=lambda x: x[0])
    q['stalk-root'] = st.radio(\"12) Raiz do caule (stalk-root)\", options=[('Uniforme / equal','e'),('Em forma de clava / club','c'),('Bulbosa / bulbous','b'),('Enraizada / rooted','r')], format_func=lambda x: x[0])
    q['stalk-surface-above-ring'] = st.radio(\"13) Superfície do caule acima do anel\", options=[('Lisa / smooth','s'),('Fibrosa / fibrous','f'),('Sedosa / silky','k'),('Escamosa / scaly','y')], format_func=lambda x: x[0])
    q['stalk-surface-below-ring'] = st.radio(\"14) Superfície do caule abaixo do anel\", options=[('Lisa / smooth','s'),('Fibrosa / fibrous','f'),('Escamosa / scaly','y'),('Sedosa / silky','k')], format_func=lambda x: x[0])
    q['stalk-color-above-ring'] = st.radio(\"15) Cor do caule acima do anel\", options=[('Branco / white','w'),('Cinza / gray','g'),('Rosa / pink','p'),('Marrom / brown','n'),('Bege / buff','b'),('Vermelho / red','e'),('Laranja / orange','o'),('Canela / cinnamon','c'),('Amarelo / yellow','y')], format_func=lambda x: x[0])
    q['stalk-color-below-ring'] = st.radio(\"16) Cor do caule abaixo do anel\", options=[('Branco / white','w'),('Rosa / pink','p'),('Cinza / gray','g'),('Bege / buff','b'),('Marrom / brown','n'),('Vermelho / red','e'),('Amarelo / yellow','y'),('Laranja / orange','o'),('Canela / cinnamon','c')], format_func=lambda x: x[0])
    q['veil-type'] = st.radio(\"17) Tipo de véu (veil-type)\", options=[('Parcial / partial','p')], format_func=lambda x: x[0])
    q['veil-color'] = st.radio(\"18) Cor do véu (veil-color)\", options=[('Branco / white','w'),('Marrom / brown','n'),('Laranja / orange','o'),('Amarelo / yellow','y')], format_func=lambda x: x[0])
    q['ring-number'] = st.radio(\"19) Número de anéis (ring-number)\", options=[('Um / one','o'),('Dois / two','t'),('Nenhum / none','n')], format_func=lambda x: x[0])
    q['ring-type'] = st.radio(\"20) Tipo de anel (ring-type)\", options=[('Pendente / pendant','p'),('Evanescente / evanescent','e'),('Grande / large','l'),('Expandido / flaring','f'),('Nenhum / none','n')], format_func=lambda x: x[0])
    q['spore-print-color'] = st.radio(\"21) Cor do esporo (spore-print-color)\", options=[('Preto / black','k'),('Marrom / brown','n'),('Roxo / purple','u'),('Chocolate','h'),('Branco / white','w'),('Verde / green','r'),('Laranja / orange','o'),('Amarelo / yellow','y'),('Bege / buff','b')], format_func=lambda x: x[0])
    q['population'] = st.radio(\"22) Como é a população (population)?\", options=[('Dispersa / scattered','s'),('Numerosa / numerous','n'),('Abundante / abundant','a'),('Várias / several','v'),('Solitária / solitary','y'),('Agrupada / clustered','c')], format_func=lambda x: x[0])
    q['habitat'] = st.radio(\"23) Habitat\", options=[('Urbano / urban','u'),('Gramados / grasses','g'),('Prados / meadows','m'),('Florestas / woods','d'),('Trilhas / paths','p'),('Terrenos baldios / waste','w'),('Folhas / leaves','l')], format_func=lambda x: x[0])

    submit = st.form_submit_button(\"Enviar para Análise\")

# handle submission
if submit:
    if 'model' not in st.session_state:
        st.warning(\"Treine o modelo primeiro (clique em 'Treinar modelo agora').\")
    else:
        model = st.session_state['model']
        encoders = st.session_state['encoders']
        feature_columns = st.session_state['feature_columns']
        target_le = st.session_state['target_le']

        # build raw input dict (strings / codes)
        raw = {}
        for col in feature_columns:
            # some dataset columns include hyphens; our form uses same keys
            val = q.get(col, None)
            if val is None:
                # fallback: try alternate key names (replace '_' with '-')
                val = q.get(col.replace('_','-'), None)
            raw[col] = val if val is not None else ''
        # convert to numeric vector following preprocess logic
        vec = []
        for col in feature_columns:
            val = raw[col]
            # boolean-like?
            ser_example = df[col]
            if is_boolean_like(ser_example):
                mapping_vals = {True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0, 't':1,'f':0,'y':1,'n':0,'yes':1,'no':0,'1':1,'0':0}
                vec.append(int(mapping_vals.get(val, 1 if str(ser_example.mode().iloc[0]).lower() in ['true','t','y','yes','1'] else 0)))
            else:
                # use encoder if available
                if col in encoders:
                    le = encoders[col]
                    try:
                        transformed = int(le.transform([str(val)])[0])
                    except Exception:
                        # unseen value -> map to most frequent label index (fallback 0)
                        transformed = 0
                    vec.append(transformed)
                else:
                    vec.append(0)

        arr = np.array(vec).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0] if hasattr(model, 'predict_proba') else None
        pred_label = target_le.inverse_transform([pred])[0]
        if pred_label.lower().startswith('e'):
            st.success(f\"🍽️ Previsto: COMESTÍVEL (label = {pred_label})\")
        else:
            st.error(f\"☠️ Previsto: VENENOSO (label = {pred_label})\")
        if proba is not None:
            prob_df = pd.DataFrame({'classe': target_le.classes_, 'probabilidade': proba})
            st.table(prob_df)

st.markdown(\"---\")
st.markdown(\"Arquivo de formulário gerado: `forms.html`. Você pode baixar o arquivo gerado ao lado.\")\n
