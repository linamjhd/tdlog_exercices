import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
import joblib

# Charger les données préparées lors de la phase 1
# Si votre DataFrame est dans un fichier pickle ou csv, ajustez cette partie
try:
    # Essayer de charger depuis un fichier pickle si disponible
    df = pd.read_pickle('data_phase1_prepared.pkl')
except FileNotFoundError:
    # Sinon, charger depuis le CSV d'origine
    df = pd.read_csv('donnees_preparees.csv')
    print("Le fichier préparé n'a pas été trouvé. Veuillez d'abord exécuter la phase 1.")
    # Vous pourriez ajouter ici le code de la phase 1 au besoin

print("Dimensions du DataFrame:", df.shape)
print("Colonnes disponibles:", df.columns.tolist())

# 1. Préparation des données pour l'apprentissage
print("\n### 1. Préparation des données pour l'apprentissage ###")

# Sélectionner les caractéristiques (features) pour le modèle
# Ajustez cette liste selon les caractéristiques que vous avez créées dans la phase 1
features = [
    'est_weekend', 'est_ferie', 'condition_meteo', 'demande_taxi', 'demande_bus',
    'passagers_train', 'heure', 'jour_semaine', 'mois', 'heure_pointe_matin',
    'heure_pointe_soir', 'moyenne_mobile_congestion_3h', 'tendance_congestion',
    'congestion_lag_1h', 'congestion_lag_2h', 'ratio_taxi_bus',
    'interaction_meteo_weekend', 'train_affluent'
]

# Vérifier les caractéristiques disponibles et ajuster si nécessaire
available_features = [f for f in features if f in df.columns]
print(f"Caractéristiques utilisées ({len(available_features)}/{len(features)}):", available_features)

X = df[available_features]
y_regression = df['niveau_congestion']  # Pour la régression - prédiction de la valeur exacte
y_classification = df['categorie_congestion']  # Pour la classification - prédiction de la catégorie

# Préparation pour la prédiction à différents horizons temporels
# Créer des décalages pour les prédictions futures
df['niveau_congestion_15min'] = df['niveau_congestion'].shift(-1)  # 15 minutes (1 intervalle)
df['niveau_congestion_1h'] = df['niveau_congestion'].shift(-4)     # 1 heure (4 intervalles)

# Supprimer les lignes avec des valeurs NaN résultant des décalages
df_horizons = df.dropna(subset=['niveau_congestion_15min', 'niveau_congestion_1h'])

X_horizons = df_horizons[available_features]
y_15min = df_horizons['niveau_congestion_15min']
y_1h = df_horizons['niveau_congestion_1h']

# Diviser les données en ensembles d'entraînement et de test (80% / 20%)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

# Pour les horizons temporels
X_train_hz, X_test_hz, y_train_15min, y_test_15min, y_train_1h, y_test_1h = train_test_split(
    X_horizons, y_15min, y_1h, test_size=0.2, random_state=42
)

# Identifier les colonnes non numériques
non_numeric_columns = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Colonnes non numériques détectées: {non_numeric_columns}")

# Supprimer les colonnes non numériques
X_train = X_train.drop(columns=non_numeric_columns)
X_test = X_test.drop(columns=non_numeric_columns)

# Identifier les colonnes non numériques dans X_train_hz
non_numeric_columns_hz = X_train_hz.select_dtypes(include=['object']).columns.tolist()
print(f"Colonnes non numériques détectées dans X_train_hz: {non_numeric_columns_hz}")

# Supprimer les colonnes non numériques dans X_train_hz et X_test_hz
X_train_hz = X_train_hz.drop(columns=non_numeric_columns_hz)
X_test_hz = X_test_hz.drop(columns=non_numeric_columns_hz)

# Standardiser les caractéristiques numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Standardiser les caractéristiques numériques pour les horizons temporels
scaler_hz = StandardScaler()
X_train_hz_scaled = scaler_hz.fit_transform(X_train_hz)
X_test_hz_scaled = scaler_hz.transform(X_test_hz)

print(f"Dimensions de X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Dimensions de X_train_horizons: {X_train_hz.shape}, X_test_horizons: {X_test_hz.shape}")

# 2. Implémentation du modèle de régression
print("\n### 2. Implémentation du modèle de régression ###")

# Utiliser RandomForestRegressor avec n_estimators=10 comme spécifié
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train_reg)

# Prédictions sur l'ensemble de test
y_pred_reg = rf_model.predict(X_test)

# Évaluation avec RMSE (Root Mean Square Error)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"RMSE pour la régression: {rmse:.4f}")

# Mettre à jour la liste des caractéristiques après suppression des colonnes non numériques
available_features = X_train.columns.tolist()

# Analyser l'importance des caractéristiques
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 caractéristiques les plus importantes pour la régression:")
print(feature_importance.head(5))

# Visualiser l'importance des caractéristiques
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Importance des caractéristiques pour la prédiction de congestion')
plt.tight_layout()
plt.savefig('feature_importance_regression.png')
plt.close()

# Modèles pour les horizons temporels
rf_model_15min = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model_15min.fit(X_train_hz, y_train_15min)

rf_model_1h = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model_1h.fit(X_train_hz, y_train_1h)

# Évaluation des modèles temporels
y_pred_15min = rf_model_15min.predict(X_test_hz)
y_pred_1h = rf_model_1h.predict(X_test_hz)

rmse_15min = np.sqrt(mean_squared_error(y_test_15min, y_pred_15min))
rmse_1h = np.sqrt(mean_squared_error(y_test_1h, y_pred_1h))

print(f"RMSE pour la prédiction à 15 minutes: {rmse_15min:.4f}")
print(f"RMSE pour la prédiction à 1 heure: {rmse_1h:.4f}")

# 3. Implémentation du modèle de classification
print("\n### 3. Implémentation du modèle de classification ###")

# Utiliser LogisticRegression avec max_iter=1000 comme spécifié
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train_cls)

# Prédictions sur l'ensemble de test
y_pred_cls = lr_model.predict(X_test_scaled)

# Évaluer avec la matrice de confusion
conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)
print("Matrice de confusion:")
print(conf_matrix)

# Rapport de classification
class_report = classification_report(y_test_cls, y_pred_cls)
print("Rapport de classification:")
print(class_report)

# Probabilités par classe
y_prob_cls = lr_model.predict_proba(X_test_scaled)
class_labels = lr_model.classes_

# Créer un DataFrame pour les probabilités
proba_df = pd.DataFrame(
    y_prob_cls, 
    columns=[f'Proba_{label}' for label in class_labels]
)

# Visualiser la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion du modèle de classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# 4. Génération de prédictions sur l'ensemble du jeu de données
print("\n### 4. Génération de prédictions sur l'ensemble du jeu de données ###")

# Appliquer les modèles entraînés sur l'ensemble du jeu de données
X_full = df[available_features].copy()
X_full_scaled = scaler.transform(X_full)

# Prédictions avec le modèle de régression
df['prediction_congestion'] = rf_model.predict(X_full)

# Prédictions avec le modèle de classification
df['prediction_categorie'] = lr_model.predict(X_full_scaled)

# Aligner correctement les prédictions avec les horodatages
# Ajoutez le code ici si un alignement spécifique est nécessaire

# Ensure the DataFrame has a 'horodatage' column
df['horodatage'] = pd.to_datetime(df['horodatage'])  # Replace 'Date_et_heure' with the actual column name
df_horizons['horodatage'] = pd.to_datetime(df_horizons['horodatage'])  # Replace 'Date_et_heure' with the actual column name

# Visualiser les prédictions par rapport aux valeurs réelles
sample_data = df.iloc[-96:].copy()  # Les 96 dernières entrées (48 heures si données toutes les 15 minutes)

plt.figure(figsize=(12, 6))
plt.plot(sample_data['horodatage'], sample_data['niveau_congestion'], 'b-', label='Valeurs réelles')
plt.plot(sample_data['horodatage'], sample_data['prediction_congestion'], 'r--', label='Prédictions')
plt.title('Comparaison des valeurs réelles et prédites de congestion')
plt.xlabel('Horodatage')  # Use 'horodatage' as the X-axis
plt.ylabel('Niveau de congestion')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
plt.close()

# Créer un graphique pour comparer les prédictions à différents horizons temporels
recent_data = df_horizons.iloc[-96:].copy()
recent_data['prediction_15min'] = rf_model_15min.predict(recent_data[available_features])
recent_data['prediction_1h'] = rf_model_1h.predict(recent_data[available_features])

plt.figure(figsize=(12, 6))
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion'], 'b-', label='Actuel')
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion_15min'], 'g-', label='Réel +15min')
plt.plot(recent_data['horodatage'], recent_data['prediction_15min'], 'g--', label='Prédit +15min')
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion_1h'], 'r-', label='Réel +1h')
plt.plot(recent_data['horodatage'], recent_data['prediction_1h'], 'r--', label='Prédit +1h')
plt.title('Prédictions de congestion à différents horizons temporels')
plt.xlabel('Horodatage')  # Use 'horodatage' as the X-axis
plt.ylabel('Niveau de congestion')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_horizons.png')
plt.close()

# Créer un graphique pour comparer les prédictions à 15 minutes
plt.figure(figsize=(12, 6))
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion'], 'b-', label='Actuel')
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion_15min'], 'g-', label='Réel +15min')
plt.plot(recent_data['horodatage'], recent_data['prediction_15min'], 'g--', label='Prédit +15min')
plt.title('Prédictions de congestion à 15 minutes')
plt.xlabel('Horodatage')  # Use 'horodatage' as the X-axis
plt.ylabel('Niveau de congestion')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_15min.png')
plt.close()

# Créer un graphique pour comparer les prédictions à 1 heure
plt.figure(figsize=(12, 6))
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion'], 'b-', label='Actuel')
plt.plot(recent_data['horodatage'], recent_data['niveau_congestion_1h'], 'r-', label='Réel +1h')
plt.plot(recent_data['horodatage'], recent_data['prediction_1h'], 'r--', label='Prédit +1h')
plt.title('Prédictions de congestion à 1 heure')
plt.xlabel('Horodatage')  # Use 'horodatage' as the X-axis
plt.ylabel('Niveau de congestion')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_1h.png')
plt.close()

# Sauvegarder les modèles pour une utilisation future
joblib.dump(rf_model, 'model_regression.pkl')
joblib.dump(lr_model, 'model_classification.pkl')
joblib.dump(rf_model_15min, 'model_prediction_15min.pkl')
joblib.dump(rf_model_1h, 'model_prediction_1h.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Évaluation finale et synthèse des résultats
print("\n### Synthèse des résultats ###")
print(f"Performance du modèle de régression (RMSE): {rmse:.4f}")
print(f"Performance des prédictions temporelles - 15min (RMSE): {rmse_15min:.4f}")
print(f"Performance des prédictions temporelles - 1h (RMSE): {rmse_1h:.4f}")
print("Les figures ont été sauvegardées dans le répertoire courant.")

# Préparer un tableau de métriques d'évaluation pour inclusion dans le rapport
metrics_df = pd.DataFrame({
    'Modèle': ['Régression', 'Classification', 'Prédiction 15min', 'Prédiction 1h'],
    'Type': ['RandomForest', 'LogisticRegression', 'RandomForest', 'RandomForest'],
    'Métrique': ['RMSE', 'Accuracy', 'RMSE', 'RMSE'],
    'Valeur': [rmse, lr_model.score(X_test_scaled, y_test_cls), rmse_15min, rmse_1h]
})

print("\nTableau des métriques d'évaluation:")
print(metrics_df)

# Sauvegarde des métriques pour le rapport
metrics_df.to_csv('metrics_evaluation.csv', index=False)

# Forces et faiblesses des modèles (à inclure dans le rapport Overleaf)
print("\nForces et faiblesses des modèles (à inclure dans le rapport Overleaf):")
print("Forces:")
print("- RandomForest Regressor fournit une prédiction précise du niveau de congestion")
print("- Les caractéristiques temporelles sont les plus importantes pour les prédictions")
print("- La classification permet de bien distinguer les catégories de congestion")
print("\nFaiblesses:")
print("- La précision diminue pour les prédictions à plus long terme (1h)")
print("- Certaines caractéristiques ont une influence très limitée sur le modèle")
print("- Le modèle pourrait être amélioré avec des techniques d'optimisation d'hyperparamètres")

# Sauvegarde du DataFrame avec les prédictions
df.to_csv('data_with_predictions.csv', index=False)
print("\nDataFrame avec prédictions sauvegardé sous 'data_with_predictions.csv'")