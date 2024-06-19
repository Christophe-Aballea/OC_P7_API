from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

# Chemin absolu du répertoire courant
base_dir = os.path.dirname(os.path.abspath(__file__))
# Chemin absolu de 'run_id.txt'
run_id_path = os.path.join(base_dir, '..', 'data', 'run_id.txt')

# Vérification de l'existence du fichier run_id.txt
if not os.path.exists(run_id_path):
    raise FileNotFoundError("Le fichier 'run_id.txt' est introuvable. Il doit contenir la run_id MLflow du modèle.")

# Récupération du run_id MLflow
with open(run_id_path, "r") as f:
    run_id = f.read().strip()

# Vérification que le serveur MLflow est en cours d'exécution
mlflow_tracking_uri = "http://127.0.0.1:5001"
try:
    response = requests.get(mlflow_tracking_uri)
    if response.status_code != 200:
        raise ConnectionError(f"Impossible de se connecter au serveur MLflow à {mlflow_tracking_uri}. Lancement du serveur : 'mlflow ui --port 5001'.")
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Erreur de connexion au serveur MLflow : {e}")

mlflow.set_tracking_uri(mlflow_tracking_uri)

# Chargement du modèle
model_uri = f'runs:/{run_id}/LogisticRegressionModel'
try:
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Chemin absolu de 'scaler.pkl'
scaler_path = os.path.join(base_dir, '..', 'data', 'scaler.pkl')

# Vérification de l'existence du fichier scaler.pkl
if not os.path.exists(scaler_path):
    raise FileNotFoundError("Le fichier 'scaler.pkl' est introuvable. Il doit contenir le scaler entraîné sur le jeu d'entraînement.")

# Chargement du scaler
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du scaler : {e}")

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
        app.run(host='127.0.0.1', port=5002)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du démarrage de l'application Flask : {e}")
