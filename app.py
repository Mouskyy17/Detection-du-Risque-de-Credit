import streamlit as st
import numpy as np

# Charger le modèle
model = joblib.load("model.joblib")

# Interface utilisateur Streamlit
st.title("Prédiction du Risque de Crédit")

st.sidebar.header("Entrez les caractéristiques du client")

# Exemple de champs pour la saisie utilisateur
age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=30)
revenu = st.sidebar.number_input("Revenu Annuel", min_value=1000, max_value=100000, value=30000)
historique_credit = st.sidebar.slider("Historique de Crédit", 0, 1, 1)

# Normalisation des entrées
input_data = np.array([[age, revenu, historique_credit]])  # Ajustez selon votre dataset
input_scaled = scaler.transform(input_data)

# Prédiction
prediction = model.predict(input_scaled)
st.write("## Résultat de la prédiction :", "Client Risqué" if prediction[0] == 1 else "Client Non Risqué")
