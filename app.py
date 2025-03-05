import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="Prédiction Client", layout="wide")

# -------------------------------
# Chargement du modèle et des données
# -------------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    # Remplacez "model.joblib" par le chemin vers votre modèle pré-entraîné sauvegardé avec joblib
    model = joblib.load("model.joblib")
    return model

model = load_model()

@st.cache
def load_data():
    # Chargez vos données pour la visualisation (ex : un fichier CSV généré dans le notebook)
    data = pd.read_csv("credit_risk_dataset.csv")
    return data

data = load_data()

# -------------------------------
# Titre et explications
# -------------------------------
st.title("Application de Prédiction Client")
st.write("Entrez les caractéristiques du client pour obtenir une prédiction en temps réel, et explorez les données avec des visualisations interactives.")

# -------------------------------
# Saisie des caractéristiques du client (Sidebar)
# -------------------------------
st.sidebar.header("Caractéristiques du client")

age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
revenu = st.sidebar.number_input("Revenu annuel (en €)", min_value=0, value=30000)
score = st.sidebar.slider("Score de crédit", min_value=300, max_value=850, value=600)

# Construction d'un DataFrame pour passer les données au modèle
client_df = pd.DataFrame({
    "age": [age],
    "revenu": [revenu],
    "score": [score]
})

# Bouton pour lancer la prédiction
if st.sidebar.button("Prédire"):
    prediction = model.predict(client_df)
    st.subheader("Résultat de la Prédiction")
    st.write(f"Le modèle prédit : **{prediction[0]}**")

st.write("---")

# -------------------------------
# Visualisations dynamiques
# -----------
