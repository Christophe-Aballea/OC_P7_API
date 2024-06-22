import streamlit as st
import pandas as pd
import requests
import json
import os

# Fonction de prédiction depuis l'API
def get_predictions(client_data):
    data_json = json.dumps({"data": client_data.drop(columns='SK_ID_CURR').values.tolist()})
    response = requests.post(
        'https://failurerepay-35fab8d4873c.herokuapp.com/predict',
        headers={'Content-Type': 'application/json'},
        data=data_json
    )
    return response.json()

# Chemin du fichier CSV contenant les données de test
base_dir = os.path.dirname(os.path.abspath(__file__))
df_test_median_path = os.path.join(base_dir, '..', '..', 'P07 - Implémentez un modèle de scoring', 'df_test_median.csv')

# Chargement des données de test
df_test = pd.read_csv(df_test_median_path, index_col=None)

# Interface Streamlit
st.title("Prédiction de défaut de remboursement")

# Sélection de l'ID client dans une liste déroulante
client_id = st.selectbox("Sélectionnez l'ID du client", df_test['SK_ID_CURR'].astype('int'))

# Filtre des données pour le client sélectionné
client_data = df_test[df_test['SK_ID_CURR'] == client_id]

# Prédiction
if not client_data.empty:
    prediction = get_predictions(client_data)
    failure_probability = prediction[0]
    credit_agreement = failure_probability < 0.51

    # Affichage des résultats
    st.write(f"ID Client: {int(client_id)}")
    st.write(f"Probabilité de défaut de remboursement : {failure_probability:.2f}")
    st.write(f"Accord de crédit : {'Oui' if credit_agreement else 'Non'}")
else:
    st.write("Sélectionner un ID client valide.")