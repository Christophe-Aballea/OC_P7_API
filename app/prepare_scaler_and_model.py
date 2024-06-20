import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os


# Chemins des fichiers
base_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(base_dir, '..', 'data', 'scaler.pkl')
model_path = os.path.join(base_dir, '..', 'data', 'model.pkl')
df_train_median_path = os.path.join(base_dir, '..', 'data', 'df_train_median.csv')

# Chargement des données d'entraînement
df_train = pd.read_csv(df_train_median_path)

# Séparation X_train, y_train
X_train = df_train.drop(columns=['TARGET', 'SK_ID_CURR'])
y_train = df_train['TARGET']

# Mise à l'échelle StandardScaler
scaler = StandardScaler().fit(X_train.values)
X_train = scaler.transform(X_train.values)

# Sauvegarde du scaler
joblib.dump(scaler, scaler_path)

# Entraînement du modèle
lr_model = LogisticRegression(C=5e-05, class_weight='balanced', max_iter=10000, solver='lbfgs')
lr_model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné
joblib.dump(lr_model, model_path)
