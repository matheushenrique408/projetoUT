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
        st.error(f"Arquivo '{path}' não encontrado. Coloque o CSV na pasta do projeto.")
        st.stop()
    return pd.read_csv(path)

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
            mappin
