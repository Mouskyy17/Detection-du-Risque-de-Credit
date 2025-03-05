import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="Prédiction Client", layout="wide")

# -------------------------------
# Chargement du modèle et des données
# -------------------------------
def load_model():
    # Remplacez "model.joblib" par le chemin vers votre modèle pré-entraîné sauvegardé avec joblib
    model = joblib.load("model.joblib")
    return model

model = load_model()

def load_data():
    # Chargez vos données pour la visualisation (par exemple un fichier CSV généré dans le notebook)
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

person_age = st.sidebar.slider("Âge", min_value=18, max_value=100, value=30)
person_income = st.sidebar.number_input("Revenu annuel (en €)", min_value=0, value=30000)
person_home_ownership = st.sidebar.selectbox("Propriété du domicile", options=["OWN", "MORTGAGE", "RENT", "OTHER"])
person_emp_length = st.sidebar.number_input("Durée d'emploi (en années)", min_value=0, value=5)
loan_intent = st.sidebar.selectbox("Intention de prêt", options=["personal", "credit_card", "home_improvement", "small_business", "debt_consolidation"])
loan_grade = st.sidebar.selectbox("Grade du prêt", options=["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Montant du prêt", min_value=0, value=10000)
loan_int_rate = st.sidebar.slider("Taux d'intérêt du prêt (%)", min_value=0.0, max_value=30.0, value=10.0)
loan_percent_income = st.sidebar.slider("Pourcentage du revenu", min_value=0.0, max_value=100.0, value=20.0)
cb_person_default_on_file = st.sidebar.selectbox("Défaut de paiement enregistré", options=["Y", "N"])
cb_person_cred_hist_length = st.sidebar.number_input("Longueur de l'historique de crédit (en années)", min_value=0, value=5)

# -------------------------------
# Création du DataFrame client_df
# -------------------------------
client_df = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [person_home_ownership],
    'person_emp_length': [person_emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

# -------------------------------
# Prédiction en temps réel
# -------------------------------
if st.sidebar.button("Prédire"):
    prediction = model.predict(client_df)
    st.subheader("Résultat de la Prédiction")
    st.write(f"Le modèle prédit : **{prediction[0]}**")

st.write("---")

# -------------------------------
# Visualisations dynamiques
# -------------------------------
st.subheader("Exploration des Données")

# Exemple 1 : Histogramme de la répartition des âges
fig_age = px.histogram(data, x="person_age", nbins=20, title="Répartition des âges")
st.plotly_chart(fig_age, use_container_width=True)

# Exemple 2 : Nuage de points Revenu vs Montant du prêt
fig_scatter = px.scatter(data, x="person_income", y="loan_amnt", color="person_age",
                         title="Revenu vs Montant du Prêt",
                         labels={"person_income": "Revenu annuel (€)", "loan_amnt": "Montant du prêt"})
st.plotly_chart(fig_scatter, use_container_width=True)
