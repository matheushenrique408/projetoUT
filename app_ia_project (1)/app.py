
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import joblib

# -----------------------
# CONFIGURAÇÕES / PATHS
# -----------------------
DATA_PATH = "data.csv"
MODEL_PATH = "rf_model.joblib"

# -----------------------
# CARREGAMENTO E CACHE
# -----------------------
@st.cache_data
def get_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def train_model(data):
    selected_features = ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'PTRATIO']
    x = data[selected_features]
    y = data['MEDV']
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x, y)
    joblib.dump(rf, MODEL_PATH)
    return rf

def load_model_or_train(data):
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception:
            return train_model(data)
    else:
        return train_model(data)

# -----------------------
# INÍCIO DO APP
# -----------------------
st.set_page_config(page_title="AppIA - Previsão de Preço de Imóveis", layout="wide")

st.title('AppIA - Previsão de Preço de Imóveis')

st.markdown(
    'Este é um AppIA treinado para prover preços de imóveis da cidade de Boston. '
    'Autores: Renan Aprigio Dias de Moura e Matheus Henrique Araújo Miranda.'
)

try:
    data = get_data()
except FileNotFoundError:
    st.error(f"Arquivo '{DATA_PATH}' não encontrado. Coloque o CSV na pasta do projeto.")
    st.stop()

model = load_model_or_train(data)

st.subheader('Amostra dos dados - selecione os atributos da tabela')

defaultcols = ['RM', 'PTRATIO', 'CRIM', 'MEDV']
cols = st.multiselect('Atributos', data.columns.tolist(), default=defaultcols)
st.dataframe(data[cols].head(10))

st.subheader('Distribuição de imóveis por preço')

faixa_valores = st.slider(
    'Selecione a faixa de preço (MEDV)',
    float(data.MEDV.min()),
    float(data.MEDV.max() if data.MEDV.max() > 150 else 150.0),
    (10.0, 100.0)
)

dados_filtrados = data[data['MEDV'].between(left=faixa_valores[0], right=faixa_valores[1])]

fig = px.histogram(dados_filtrados, x='MEDV', nbins=100, title='Distribuição de Preços')
fig.update_xaxes(title='MEDV')
fig.update_yaxes(title='Total imóveis')
st.plotly_chart(fig, use_container_width=True)

st.sidebar.subheader('Entre com as informações do imóvel a ser avaliado')

CRIM = st.sidebar.number_input('Taxa de criminalidade (CRIM)', value=float(data.CRIM.mean()))
INDUS = st.sidebar.number_input('Proporção de hectares de negócios (INDUS)', value=float(data.INDUS.mean()))
NOX = st.sidebar.number_input('Concentração de óxido nítrico (NOX)', value=float(data.NOX.mean()))
RM = st.sidebar.number_input('Número médio de quartos por residência (RM)', value=float(data.RM.mean()))
AGE = st.sidebar.number_input(
    'Proporção de unidades ocupadas por proprietários construídas antes de 1940 (AGE)',
    value=float(data.AGE.mean())
)
PTRATIO = st.sidebar.number_input('Índice de alunos por professor (PTRATIO)', value=float(data.PTRATIO.mean()))

CHAS = st.sidebar.selectbox('Faz limite com o rio (CHAS)?', ('Não', 'Sim'))
CHAS = 1 if CHAS == 'Sim' else 0

btn_predict = st.sidebar.button('Realizar Previsão')

if btn_predict:
    try:
        features = [[CRIM, INDUS, CHAS, NOX, RM, AGE, PTRATIO]]
        predicted = model.predict(features)
        preco = round(predicted[0] * 1000, 2)
        st.subheader('O valor previsto para o imóvel é:')
        st.write(f'US $ {preco}')
    except Exception as e:
        st.error(f"Erro ao prever: {e}")

st.markdown(
    """---
**Nota para integração da sua IA**:
Se você tiver um modelo/IA já treinada, substitua a função `train_model`
ou modifique `load_model_or_train` para carregar seu artefato (por exemplo, um `.pt`, `.pkl` ou outro).
Procure por `# SE VOCÊ TIVER UMA IA PRONTA` no topo do arquivo.
"""
)
