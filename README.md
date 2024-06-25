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
└── build_deploy.yml : Workflow GitHub Actions pour le déploiement sur Heroku  

app/  
├── init.py   
├── failure_predict.py : Script Flask pour l'API de prédiction   
└── prepare_scaler_and_model.py : Script pour entraîner et sauvegarder le scaler et le modèle  

data/  
├── df_train_median.csv : Données d'entraînement  
├── model.pkl : Modèle entraîné  
└── scaler.pkl : Scaler entraîné  

notebooks/  
├── Modelisation_FI.ipynb : Notebook modélisation, feature importance et analyse data-drift  
├── Preprocessing_feature_engineering.ipynb : Notebook prétraitement des données et feature ingineering  
└── test_API.ipynb : Notebook pour tester l'API  

streamlit/  
└── app.py : Application Streamlit pour tester l'API  

tests/  
├── init.py  
└── test_predict_api.py : Tests unitaires pour l'API  

.gitignore : Fichiers et répertoires à ignorer par Git  
.gitattributes : Configuration de Git LFS  
.slugignore : Fichiers et répertoires à ignorer par Heroku  
Procfile : Configuration Heroku  
README.md : Documentation du projet
requirements.txt : Dépendances Python  

## Prérequis  
- Python 3.11  
- Pip (gestionnaire de paquets Python)  
- Git  
- [Streamlit](https://streamlit.io/)    

## Installation  
Cloner le dépôt GitHub et installer les dépendances :  

```bash  
git clone https://github.com/Christophe-Aballea/OC_P7_API.git  
cd OC_P7_API  
pip install -r requirements.txt  
```

## Préparation du Modèle
Exécuter le script `prepare_scaler_and_model.py` pour entraîner le modèle et sauvegarder le scaler et le modèle :
```bash  
python app/prepare_scaler_and_model.py
```
Cette opération n'est pas nécessaire manuellement, elle est automatiquement réalisée par le workflow Github Actions.  

## Déploiement  
### Déploiement sur Heroku  
Le déploiement est automatisé avec GitHub Actions. Chaque push sur la branche main déclencher le workflow de déploiement.

Les secrets suivants doivent être configurés dans les paramètres du dépôt GitHub :  

* `HEROKU_API_KEY` : Clé API Heroku  
* `HEROKU_APP_NAME` : Nom de l'application Heroku  
* `HEROKU_EMAIL` : Adresse e-mail associée au compte Heroku  

### Application Streamlit  
Exécuter l'applisation streamlit localement :  
```bash  
streamlit run streamlit/app  
```

Cette application permet de sélectionner l'ID d'un client du jeu de données de test et d'obtenir :  
- La probabilité qu'il soit en défaut de remboursement si on lui accorde un crédit
- L'accord ou non du crédit en fonction du seuil de classification

Cette application doit être exécutée localement car elle n'est pas déployée sur Heroku.  
En revanche elle utilise l'API déployée, et ne peut donc être utilisée qu'après le déploiement.  

## Tests  
### Tests unitaires  
Exécutables par la commande :  
```bash  
python -m unittest discover tests  
```

Ils sont inclus en fin de workflow Github Actions et vérifient que :  
- Le code 'status' de la requête POST est bien 200  
- La réponse est bien une liste  
- La liste reçue ne contient bien qu'un seul item  
- L'item est bien de type 'float'  
- Une requête avec des données invalides renvoie bien le code status 400
- Une requête sans données renvoie bien le code status 400
- Une requête sans contenu JSON renvoie bien le code status 400

### Notebook  
Le notebook `notebooks\test_API.ipynb` extrait quelques clients du jeu de données de test et utilise l'API déployée pour effectuer les prédictions.  
