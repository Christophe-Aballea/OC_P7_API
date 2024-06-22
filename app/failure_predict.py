from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

# Chemins des fichiers scaler et model
base_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(base_dir, '..', 'data', 'scaler.pkl')
model_path = os.path.join(base_dir, '..', 'data', 'model.pkl')

# Vérification de l'existence du fichier scaler.pkl
if not os.path.exists(scaler_path):
    raise FileNotFoundError("Le fichier 'scaler.pkl' est introuvable. Il doit contenir le scaler entraîné sur le jeu d'entraînement.")

# Chargement du scaler
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du scaler : {e}")

# Chargement du modèle
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")


# Fonction de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Le contenu doit être au format JSON"}), 400

    try:
        data = request.json
        data_df = pd.DataFrame(data['data'])
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la lecture des données : {e}"}), 400

    try:
        data_df_scaled = scaler.transform(data_df)
        predictions = model.predict_proba(data_df_scaled)[:, 1]
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction : {e}"}), 500


# Déploiement de l'API
if __name__ == '__main__':
    try:
        app.run()
    except Exception as e:
        raise RuntimeError(f"Erreur lors du démarrage de l'application Flask : {e}")
