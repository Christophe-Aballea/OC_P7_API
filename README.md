# Modèle de Scoring de Crédit  

Ce projet entraîne et déploie un modèle de scoring de crédit sur la plateforme Heroku, sous forme d'une API.  
Le modèle prédit la probabilité de défaut de remboursement d'un client. Ajustée au seuil de classification, cette prédiction permet d'indiquer si la demande de crédit peut être accordée ou non.  

Un `git push` déclenche :  
- fit, enregistrement et transform d'un StandardScaler sur les données d'entraînement  
- Entraînement et enregistrement du modèle sur les données d'entraînement mises à l'échelle  
- Déploiement sur Heroku  
- Tests unitaires sur l'API déployée  

## Structure du Projet  
.github/workflows/  
└── build_deploy.yml # Workflow GitHub Actions pour le déploiement sur Heroku  

app/  
├── init.py   
├── failure_predict.py # Script Flask pour l'API de prédiction   
└── prepare_scaler_and_model.py # Script pour entraîner et sauvegarder le scaler et le modèle  

data/  
├── df_train_median.csv # Données d'entraînement  
├── model.pkl # Modèle entraîné  
└── scaler.pkl # Scaler entraîné  

notebooks/  
├── Modelisation_FI.ipynb # Notebook pour la modélisation, le feature importance et l'analyse data-drift  
├── Preprocessing_feature_engineering.ipynb # Notebook pour le prétraitement des données et le feature ingineering  
└── test_API.ipynb # Notebook pour tester l'API  

streamlit/  
└── app.py # Application Streamlit pour tester l'API  

tests/  
├── init.py  
└── test_predict_api.py # Tests unitaires pour l'API  

.gitignore # Fichiers et répertoires à ignorer par Git  
.gitattributes # Configuration de Git LFS  
.slugignore # Fichiers et répertoires à ignorer par Heroku  
Procfile # Configuration Heroku  
requirements.txt # Dépendances Python  

## Prérequis  

- Python 3.11  
- Pip (gestionnaire de paquets Python)  
- Git  
- [Streamlit](https://streamlit.io/)    

## Installation  

Cloner ce dépôt GitHub et installer les dépendances :  

```bash  
git clone https://github.com/Christophe-Aballea/OC_P7_API.git  
cd OC_P7_API  
pip install -r requirements.txt```  



