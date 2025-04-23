import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Configuration pour une meilleure visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (14, 8)

# 1. Chargement et nettoyage des données
def charger_et_nettoyer_donnees(chemin_fichier):
    """
    Charge et nettoie les données du fichier CSV.
    """
    print("Chargement des données...")
    # Chargement des données
    df = pd.read_csv(chemin_fichier)
    
    # Conversion de l'horodatage en datetime
    df['horodatage'] = pd.to_datetime(df['horodatage'])
    
    # Vérification des valeurs manquantes
    print(f"Valeurs manquantes par colonne:\n{df.isnull().sum()}")
    
    # Traitement des valeurs manquantes pour id_train
    # D'après la description, id_train est vide s'il n'y a pas d'arrivée de train
    if 'id_train' in df.columns:
        df['presence_train'] = (~df['id_train'].isna()).astype(int)
    
    # Vérification des doublons
    doublons = df.duplicated().sum()
    print(f"Nombre de doublons: {doublons}")
    if doublons > 0:
        df = df.drop_duplicates()
        print("Doublons supprimés.")
    
    # Vérification des distributions pour détecter les anomalies
    print("\nStatistiques descriptives des variables numériques:")
    print(df.describe())
    
    return df

# 2. Extraction de caractéristiques temporelles
def extraire_caracteristiques_temporelles(df):
    """
    Extrait des caractéristiques temporelles à partir de l'horodatage.
    """
    print("\nExtraction des caractéristiques temporelles...")
    
    # Extraction de l'heure, du jour de la semaine, du mois
    df['heure'] = df['horodatage'].dt.hour
    df['minute'] = df['horodatage'].dt.minute
    df['jour_semaine'] = df['horodatage'].dt.dayofweek  # 0=Lundi, 6=Dimanche
    df['jour_mois'] = df['horodatage'].dt.day
    df['mois'] = df['horodatage'].dt.month
    
    # Nom du jour de la semaine
    jours = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 
             4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}
    df['nom_jour'] = df['jour_semaine'].map(jours)
    
    # Heures de pointe (matin: 7h-9h, soir: 17h-19h)
    df['heure_pointe_matin'] = ((df['heure'] >= 7) & (df['heure'] < 9)).astype(int)
    df['heure_pointe_soir'] = ((df['heure'] >= 17) & (df['heure'] < 19)).astype(int)
    df['heure_pointe'] = ((df['heure_pointe_matin'] == 1) | (df['heure_pointe_soir'] == 1)).astype(int)
    
    # Période de la journée
    conditions = [
        (df['heure'] >= 5) & (df['heure'] < 12),
        (df['heure'] >= 12) & (df['heure'] < 17),
        (df['heure'] >= 17) & (df['heure'] < 22),
        (df['heure'] >= 22) | (df['heure'] < 5)
    ]
    choices = ['Matin', 'Après-midi', 'Soir', 'Nuit']
    df['periode_jour'] = np.select(conditions, choices, default='Autre')
    
    # Saisons
    saisons = {
        1: 'Hiver', 2: 'Hiver', 3: 'Printemps', 4: 'Printemps', 
        5: 'Printemps', 6: 'Été', 7: 'Été', 8: 'Été', 
        9: 'Automne', 10: 'Automne', 11: 'Automne', 12: 'Hiver'
    }
    df['saison'] = df['mois'].map(saisons)
    
    # Conversion condition_meteo à partir de code_meteo si nécessaire
    if 'code_meteo' in df.columns and 'condition_meteo' not in df.columns:
        meteo_mapping = {0: 'Soleil', 1: 'Nuageux', 2: 'Pluvieux', 3: 'Neigeux'}
        df['condition_meteo'] = df['code_meteo'].map(meteo_mapping)
    
    # Conversion flux_direction à partir de code_flux_direction si nécessaire
    if 'code_flux_direction' in df.columns and 'flux_direction' not in df.columns:
        flux_mapping = {0: 'sortant', 1: 'neutre', 2: 'entrant'}
        df['flux_direction'] = df['code_flux_direction'].map(flux_mapping)
    
    return df

# 3. Création de caractéristiques avancées
def creer_caracteristiques_avancees(df):
    """
    Crée des caractéristiques avancées pour l'analyse et la modélisation.
    """
    print("\nCréation de caractéristiques avancées...")
    
    # Tri par horodatage pour calculer correctement les moyennes mobiles
    df = df.sort_values('horodatage')
    
    # Calcul de la moyenne mobile de congestion sur 3 heures (12 intervalles de 15 min)
    df['moyenne_mobile_congestion_3h'] = df['niveau_congestion'].rolling(window=12, min_periods=1).mean()
    
    # Création de variables de tendance
    df['tendance_congestion'] = df['niveau_congestion'] - df['moyenne_mobile_congestion_3h']
    
    # Ajout de décalages temporels (lag)
    df['congestion_lag_1h'] = df['niveau_congestion'].shift(4)  # 4 intervalles de 15 min = 1h
    df['congestion_lag_2h'] = df['niveau_congestion'].shift(8)  # 8 intervalles de 15 min = 2h
    
    # Remplacer les NaN des variables décalées par la moyenne
    df['congestion_lag_1h'].fillna(df['niveau_congestion'].mean(), inplace=True)
    df['congestion_lag_2h'].fillna(df['niveau_congestion'].mean(), inplace=True)
    
    # Création d'une variable catégorielle de congestion
    conditions = [
        (df['niveau_congestion'] >= 0) & (df['niveau_congestion'] < 3),
        (df['niveau_congestion'] >= 3) & (df['niveau_congestion'] < 6),
        (df['niveau_congestion'] >= 6) & (df['niveau_congestion'] <= 10)
    ]
    choices = ['Faible', 'Moyen', 'Élevé']
    df['categorie_congestion'] = np.select(conditions, choices, default='Inconnu')
    
    # Encodage numérique de la catégorie de congestion pour la modélisation
    categorie_mapping = {'Faible': 0, 'Moyen': 1, 'Élevé': 2}
    df['code_categorie_congestion'] = df['categorie_congestion'].map(categorie_mapping)
    
    # Ratio de demande taxi/bus
    if 'demande_taxi' in df.columns and 'demande_bus' in df.columns:
        df['ratio_taxi_bus'] = df['demande_taxi'] / df['demande_bus'].replace(0, 0.1)
        
    # Variables d'interaction
    if 'code_meteo' in df.columns and 'est_weekend' in df.columns:
        df['interaction_meteo_weekend'] = df['code_meteo'] * df['est_weekend']
    
    # Créer une variable pour indiquer les arrivées de train avec beaucoup de passagers
    if 'passagers_train' in df.columns and 'presence_train' in df.columns:
        # Calculer le seuil pour "beaucoup de passagers" (75e percentile des valeurs non nulles)
        seuil_passagers = df[df['passagers_train'] > 0]['passagers_train'].quantile(0.75)
        df['train_affluent'] = ((df['presence_train'] == 1) & 
                                (df['passagers_train'] >= seuil_passagers)).astype(int)
    
    return df

# 4. Analyse exploratoire des données
def analyse_exploratoire(df):
    """
    Réalise une analyse exploratoire des données avec visualisations.
    """
    print("\nAnalyse exploratoire des données...")
    
    # Création d'un dossier pour les visualisations
    import os
    if not os.path.exists('visualisations'):
        os.makedirs('visualisations')
    
    # 1. Tendance quotidienne de la congestion
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df, x='heure', y='niveau_congestion', hue='nom_jour', ci=None)
    plt.title('Tendance quotidienne de la congestion par jour de la semaine')
    plt.xlabel('Heure de la journée')
    plt.ylabel('Niveau de congestion moyen')
    plt.legend(title='Jour')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.savefig('visualisations/tendance_quotidienne_congestion.png')
    plt.close()
    
    # 2. Relation entre conditions météorologiques et congestion
    if 'condition_meteo' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='condition_meteo', y='niveau_congestion')
        plt.title('Impact des conditions météorologiques sur la congestion')
        plt.xlabel('Condition météorologique')
        plt.ylabel('Niveau de congestion')
        plt.grid(True)
        plt.savefig('visualisations/meteo_congestion.png')
        plt.close()
    
    # 3. Impact des arrivées de trains sur les niveaux de congestion
    if 'presence_train' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Calculer la congestion moyenne avec/sans arrivée de train
        train_effect = df.groupby(['heure', 'presence_train'])['niveau_congestion'].mean().unstack()
        
        # Vérifier que les deux colonnes existent (présence = 0 et présence = 1)
        if 0 in train_effect.columns and 1 in train_effect.columns:
            plt.plot(train_effect.index, train_effect[0], 
                     label='Sans arrivée de train', marker='o')
            plt.plot(train_effect.index, train_effect[1], 
                     label='Avec arrivée de train', marker='x')
            
            plt.title('Impact des arrivées de trains sur la congestion')
            plt.xlabel('Heure de la journée')
            plt.ylabel('Niveau de congestion moyen')
            plt.legend()
            plt.xticks(range(0, 24))
            plt.grid(True)
            plt.savefig('visualisations/impact_trains_congestion.png')
            plt.close()
    
    # 4. Matrice de corrélation entre les variables numériques
    plt.figure(figsize=(16, 14))
    
    # Sélection des colonnes numériques pertinentes (exclure certaines colonnes dérivées)
    cols_to_exclude = ['code_categorie_congestion', 'heure_pointe_matin', 
                       'heure_pointe_soir', 'heure_pointe', 'train_affluent']
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col not in cols_to_exclude]
    
    # Calculer la matrice de corrélation
    correlation_matrix = df[numeric_columns].corr()
    
    # Créer une heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation des variables numériques')
    plt.tight_layout()
    plt.savefig('visualisations/matrice_correlation.png')
    plt.close()
    
    # 5. Distribution de la congestion par jour de la semaine et par période
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='nom_jour', y='niveau_congestion', hue='periode_jour')
    plt.title('Distribution de la congestion par jour et période')
    plt.xlabel('Jour de la semaine')
    plt.ylabel('Niveau de congestion')
    plt.legend(title='Période de la journée')
    plt.grid(True)
    plt.savefig('visualisations/distribution_congestion_jour_periode.png')
    plt.close()
    
    # 6. Analyse des tendances saisonnières
    plt.figure(figsize=(14, 8))
    monthly_data = df.groupby('mois')['niveau_congestion'].mean().reset_index()
    sns.barplot(data=monthly_data, x='mois', y='niveau_congestion')
    plt.title('Tendances saisonnières de la congestion')
    plt.xlabel('Mois')
    plt.ylabel('Niveau de congestion moyen')
    plt.xticks(range(12), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                           'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'])
    plt.grid(True)
    plt.savefig('visualisations/tendances_saisonnieres.png')
    plt.close()
    
    return df

def visualiser_detail_pic_demande(df):
    """
    Visualise the detailed demand peak after a train arrival at 08:00.
    """
    # Filtrer les données pour un train spécifique arrivé à 08:00
    # On suppose que la date est incluse dans l'horodatage, donc on filtre sur l'heure
    train_data = df[(df['horodatage'].dt.hour == 8) & 
                    (df['horodatage'].dt.minute == 0) & 
                    (df['presence_train'] == 1)]
    
    if len(train_data) == 0:
        print("Aucune donnée d'arrivée de train à 08:00 trouvée.")
        return
    
    # Récupérer l'horodatage exact de l'arrivée du train
    date_arrivee = train_data['horodatage'].iloc[0]
    
    # Récupérer le nombre de passagers
    passagers = train_data['passagers_train'].iloc[0]
    
    # Créer une plage horaire de 2 heures après l'arrivée du train
    debut = date_arrivee
    fin = debut + pd.Timedelta(hours=2)
    
    # Filtrer les données pour cette période
    periode_data = df[(df['horodatage'] >= debut) & (df['horodatage'] <= fin)]
    
    # Créer le graphique
    plt.figure(figsize=(14, 8))
    
    # Tracer les lignes de demande de taxi et de bus
    plt.plot(periode_data['horodatage'], periode_data['demande_taxi'], 
             marker='o', color='darkgreen', label='Demande de taxis')
    plt.plot(periode_data['horodatage'], periode_data['demande_bus'], 
             marker='o', color='darkred', label='Demande de bus')
    
    # Ajouter une ligne verticale pour indiquer l'arrivée du train
    plt.axvline(x=date_arrivee, color='red', linestyle='--', 
                label=f'Arrivée du train ({passagers} passagers)')
    
    # Formater l'axe x pour afficher uniquement les heures:minutes
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Ajouter les labels et le titre
    plt.title(f"Détail d'un pic de demande après l'arrivée d'un train à {date_arrivee.strftime('%H:%M')}")
    plt.xlabel('Heure')
    plt.ylabel('Nombre de personnes par 15 min')
    plt.legend()
    plt.grid(True)
    
    # Ajuster les limites de l'axe y pour correspondre à l'image
    plt.ylim(0, 550)
    
    plt.savefig('visualisations/detail_pic_demande.png')
    plt.close()
    
    print("Graphique 'Détail d'un pic de demande' sauvegardé.")

# Fonction pour créer la Figure 2: Effet des arrivées de train sur la demande de transport
def visualiser_effet_arrivees_train(df, date_specifique=None):
    """
    Crée un graphique montrant l'effet des arrivées de train sur la demande de transport.
    Reproduit l'Image 2 du document.
    """
    # Si une date spécifique est fournie, filtrer les données pour cette date
    if date_specifique:
        df_jour = df[df['horodatage'].dt.date == date_specifique]
    else:
        # Sinon, prendre le dernier jour disponible dans les données
        derniere_date = df['horodatage'].dt.date.max()
        df_jour = df[df['horodatage'].dt.date == derniere_date]
        date_specifique = derniere_date
    
    # Créer le graphique
    plt.figure(figsize=(16, 8))
    
    # Tracer les lignes de demande de taxi et de bus
    plt.plot(df_jour['horodatage'], df_jour['demande_taxi'], 
             color='darkgreen', label='Demande de taxis')
    plt.plot(df_jour['horodatage'], df_jour['demande_bus'], 
             color='darkorange', label='Demande de bus')
    
    # Ajouter des lignes verticales pour chaque arrivée de train
    arrivees_train = df_jour[df_jour['presence_train'] == 1]
    
    for idx, train in arrivees_train.iterrows():
        plt.axvline(x=train['horodatage'], color='red', linestyle='--')
        # Ajouter le nombre de passagers au-dessus de chaque ligne
        plt.text(train['horodatage'], 550, f"{int(train['passagers_train'])} pass.",
                 rotation=90, verticalalignment='top', horizontalalignment='center',
                 fontsize=9)
    
    # Formater l'axe x pour afficher uniquement les heures
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
    
    # Ajouter les labels et le titre
    date_str = date_specifique.strftime('%Y-%m-%d')
    plt.title(f"Effet des arrivées de train sur la demande de transport - {date_str}")
    plt.xlabel('Heure')
    plt.ylabel('Nombre de personnes par 15 min')
    plt.legend()
    plt.grid(True)
    
    # Ajuster les limites de l'axe y pour correspondre à l'image
    plt.ylim(0, 550)
    
    plt.savefig('visualisations/effet_arrivees_train.png')
    plt.close()
    
    print(f"Graphique 'Effet des arrivées de train' pour le {date_str} sauvegardé.")

# Fonction pour créer la Figure 3: Demande de transport par heure de la journée
def visualiser_demande_transport_journee(df):
    """
    Crée un graphique montrant la demande de transport par heure pour les jours de semaine et weekend.
    Reproduit l'Image 3 du document.
    """
    # Créer le graphique
    plt.figure(figsize=(16, 8))
    
    # Calcul des moyennes horaires pour jours de semaine et weekend
    df_semaine = df[df['est_weekend'] == 0]
    df_weekend = df[df['est_weekend'] == 1]
    
    # Grouper par heure pour calculer les moyennes
    demande_taxi_semaine = df_semaine.groupby(df_semaine['horodatage'].dt.hour)['demande_taxi'].mean()
    demande_bus_semaine = df_semaine.groupby(df_semaine['horodatage'].dt.hour)['demande_bus'].mean()
    demande_taxi_weekend = df_weekend.groupby(df_weekend['horodatage'].dt.hour)['demande_taxi'].mean()
    demande_bus_weekend = df_weekend.groupby(df_weekend['horodatage'].dt.hour)['demande_bus'].mean()
    
    # Tracer les lignes de demande
    plt.plot(demande_taxi_semaine.index, demande_taxi_semaine.values, 
             color='darkblue', label='Taxis (semaine)', linewidth=2)
    plt.plot(demande_bus_semaine.index, demande_bus_semaine.values, 
             color='darkorange', label='Bus (semaine)', linewidth=2)
    plt.plot(demande_taxi_weekend.index, demande_taxi_weekend.values, 
             color='darkblue', linestyle='--', label='Taxis (weekend)', linewidth=2)
    plt.plot(demande_bus_weekend.index, demande_bus_weekend.values, 
             color='darkorange', linestyle='--', label='Bus (weekend)', linewidth=2)
    
    # Ajouter des zones colorées pour les heures de pointe
    plt.axvspan(7, 9, alpha=0.2, color='yellow', label='Heure de pointe du matin')
    plt.axvspan(17, 19, alpha=0.2, color='orange', label='Heure de pointe du soir')
    
    # Ajouter les labels et le titre
    plt.title('Demande de transport par heure de la journée')
    plt.xlabel('Heure')
    plt.ylabel('Demande moyenne de transport')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Ajuster les axes
    plt.xlim(0, 23)
    plt.ylim(0, 450)
    plt.xticks(range(0, 24))
    
    plt.savefig('visualisations/demande_transport_journee.png')
    plt.close()
    
    print("Graphique 'Demande de transport par heure de la journée' sauvegardé.")

# Fonction pour exécuter les trois visualisations
def generer_visualisations_specifiques(df):
    """
    Génère les trois visualisations principales demandées.
    """
    print("Génération des visualisations spécifiques...")
    
    # Création d'un dossier pour les visualisations si nécessaire
    import os
    if not os.path.exists('visualisations'):
        os.makedirs('visualisations')
    
    # Visualisation 1: Détail d'un pic de demande
    visualiser_detail_pic_demande(df)
    
    # Visualisation 2: Effet des arrivées de train sur une journée
    # On utilise la date 2024-03-05 comme dans l'image 2
    date_specifique = datetime(2024, 3, 5).date()
    visualiser_effet_arrivees_train(df, date_specifique)
    
    # Visualisation 3: Demande de transport par heure de la journée
    visualiser_demande_transport_journee(df)
    
    print("Toutes les visualisations ont été générées avec succès!")
    print("Analyse exploratoire terminée. Les visualisations ont été sauvegardées dans le dossier 'visualisations'.")
    
    return df

# 5. Identification des facteurs influençant la congestion
def identifier_facteurs_congestion(df):
    """
    Identifie et analyse spécifiquement les facteurs qui influencent le niveau de congestion.
    """
    print("\nIdentification des facteurs influençant la congestion...")
    
    # Création d'un dossier pour les visualisations si nécessaire
    import os
    if not os.path.exists('visualisations'):
        os.makedirs('visualisations')
    
    # 1. Analyse de corrélation spécifique avec la congestion
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclure certaines colonnes dérivées ou redondantes
    cols_to_exclude = ['code_categorie_congestion', 'heure_pointe_matin', 
                     'heure_pointe_soir', 'heure_pointe']
    
    numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
    
    if 'niveau_congestion' in numeric_cols:
        correlations = df[numeric_cols].corr()['niveau_congestion'].sort_values(ascending=False)
        correlations = correlations.drop('niveau_congestion')
        
        print("\nTop corrélations avec le niveau de congestion:")
        print(correlations.head(10))
        
        plt.figure(figsize=(14, 8))
        correlations.head(10).plot(kind='bar')
        plt.title('Top 10 des facteurs corrélés au niveau de congestion')
        plt.xlabel('Variables')
        plt.ylabel('Coefficient de corrélation')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualisations/top_correlations_congestion.png')
        plt.close()
    
    # 2. Analyse d'importance des variables avec Random Forest
    if 'niveau_congestion' in df.columns:
        # Sélectionner les caractéristiques et la cible
        feature_cols = [col for col in numeric_cols if col != 'niveau_congestion']
        X = df[feature_cols]
        y = df['niveau_congestion']
        
        # Entraîner un modèle Random Forest pour l'importance des variables
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        # Récupérer l'importance des variables
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        
        print("\nImportance des variables pour prédire la congestion:")
        print(importances.head(10))
        
        plt.figure(figsize=(14, 8))
        importances.head(10).plot(kind='bar')
        plt.title('Top 10 des facteurs influençant la congestion selon Random Forest')
        plt.xlabel('Variables')
        plt.ylabel('Importance relative')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualisations/importance_variables_congestion.png')
        plt.close()
    
    # 3. Analyse des modèles de pic après les arrivées de trains (Tendance clé #1)
    if 'presence_train' in df.columns and 'passagers_train' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Regrouper par nombre de passagers (en catégories) et voir l'impact sur la congestion
        df['categorie_passagers'] = pd.cut(df['passagers_train'], 
                                        bins=[0, 1, 100, 200, 500, float('inf')],
                                        labels=['Aucun', 'Très peu', 'Peu', 'Moyen', 'Nombreux'])
        
        sns.boxplot(data=df, x='categorie_passagers', y='niveau_congestion')
        plt.title('Impact du nombre de passagers sur la congestion')
        plt.xlabel('Catégorie de nombre de passagers')
        plt.ylabel('Niveau de congestion')
        plt.grid(True)
        plt.savefig('visualisations/impact_passagers_congestion.png')
        plt.close()
    
    # 4. Effets de l'heure de la journée (Tendance clé #2)
    plt.figure(figsize=(16, 8))
    hourly_data = df.groupby(['heure', 'est_weekend'])['niveau_congestion'].mean().unstack()
    
    if 0 in hourly_data.columns and 1 in hourly_data.columns:
        plt.plot(hourly_data.index, hourly_data[0], 
                label='Jour de semaine', marker='o')
        plt.plot(hourly_data.index, hourly_data[1], 
                label='Weekend', marker='x')
        
        plt.title('Effet de l\'heure et du type de jour sur la congestion')
        plt.xlabel('Heure de la journée')
        plt.ylabel('Niveau de congestion moyen')
        plt.legend()
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.savefig('visualisations/effet_heure_jour_congestion.png')
        plt.close()
    
    # 5. Relations entre demande de transport et congestion
    if 'demande_taxi' in df.columns and 'demande_bus' in df.columns:
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        sns.regplot(data=df, x='demande_taxi', y='niveau_congestion', scatter_kws={'alpha':0.3})
        plt.title('Relation entre demande de taxis et congestion')
        plt.xlabel('Demande de taxis')
        plt.ylabel('Niveau de congestion')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        sns.regplot(data=df, x='demande_bus', y='niveau_congestion', scatter_kws={'alpha':0.3})
        plt.title('Relation entre demande de bus et congestion')
        plt.xlabel('Demande de bus')
        plt.ylabel('Niveau de congestion')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualisations/relation_demande_transport_congestion.png')
        plt.close()
    
    # 6. Impact de la météo sur la congestion (Tendance clé #4)
    if 'condition_meteo' in df.columns and 'demande_taxi' in df.columns:
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df, x='condition_meteo', y='niveau_congestion')
        plt.title('Impact de la météo sur la congestion')
        plt.xlabel('Condition météorologique')
        plt.ylabel('Niveau de congestion')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        sns.boxplot(data=df, x='condition_meteo', y='demande_taxi')
        plt.title('Impact de la météo sur la demande de taxis')
        plt.xlabel('Condition météorologique')
        plt.ylabel('Demande de taxis')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualisations/impact_meteo_congestion_transport.png')
        plt.close()
    
    # 7. Variations selon le jour de la semaine (Tendance clé #5)
    plt.figure(figsize=(14, 8))
    daily_data = df.groupby('nom_jour')['niveau_congestion'].mean().reindex(
        ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    
    sns.barplot(x=daily_data.index, y=daily_data.values)
    plt.title('Niveau de congestion moyen par jour de la semaine')
    plt.xlabel('Jour de la semaine')
    plt.ylabel('Niveau de congestion moyen')
    plt.grid(True)
    plt.savefig('visualisations/variations_jour_semaine.png')
    plt.close()
    
    # 8. Tendances saisonnières (Tendance clé #6)
    plt.figure(figsize=(14, 8))
    season_data = df.groupby('saison')['niveau_congestion'].mean().reindex(
        ['Hiver', 'Printemps', 'Été', 'Automne'])
    
    sns.barplot(x=season_data.index, y=season_data.values)
    plt.title('Niveau de congestion moyen par saison')
    plt.xlabel('Saison')
    plt.ylabel('Niveau de congestion moyen')
    plt.grid(True)
    plt.savefig('visualisations/tendances_saisonnieres_simples.png')
    plt.close()
    
    # Résumé des facteurs identifiés
    print("\nRésumé des facteurs influençant la congestion:")
    
    # Créer un DataFrame pour le résumé des facteurs
    facteurs_resume = pd.DataFrame({
        'Facteur': correlations.head(10).index,
        'Coefficient_Correlation': correlations.head(10).values,
        'Importance_RF': [importances.get(factor, 0) for factor in correlations.head(10).index]
    })
    
    # Ajouter une colonne d'interprétation
    facteurs_resume['Interprétation'] = facteurs_resume.apply(
        lambda row: f"Forte corrélation ({row['Coefficient_Correlation']:.2f}) et importance RF ({row['Importance_RF']:.2f})" 
        if row['Importance_RF'] > 0.05 else 
        f"Corrélation significative ({row['Coefficient_Correlation']:.2f})", 
        axis=1
    )
    
    # Sauvegarder le résumé des facteurs
    facteurs_resume.to_csv('facteurs_influencant_congestion.csv', index=False)
    print("Résumé des facteurs influençant la congestion sauvegardé dans 'facteurs_influencant_congestion.csv'")
    
    return df

# 6. Préparation finale des données pour la modélisation
def preparer_pour_modelisation(df):
    """
    Prépare les données pour la modélisation en standardisant les variables.
    """
    print("\nPréparation des données pour la modélisation...")
    
    # Liste des caractéristiques numériques à standardiser
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclure la variable cible et les variables catégorielles
    cols_to_exclude = ['niveau_congestion', 'code_categorie_congestion']
    numeric_features = [col for col in numeric_features if col not in cols_to_exclude]
    
    # Standardisation des caractéristiques numériques
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    if numeric_features:
        df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    print("Données préparées pour la modélisation.")
    
    return df_scaled

# 7. Fonction principale pour exécuter tout le processus
def executer_phase1(chemin_fichier):
    """
    Exécute toutes les étapes de la phase 1 du projet.
    """
    # Chargement et nettoyage des données
    df = charger_et_nettoyer_donnees(chemin_fichier)
    
    # Extraction des caractéristiques temporelles
    df = extraire_caracteristiques_temporelles(df)
    
    # Création de caractéristiques avancées
    df = creer_caracteristiques_avancees(df)
    
    # Analyse exploratoire
    df = analyse_exploratoire(df)
    df = generer_visualisations_specifiques(df)
    
    # Identification des facteurs influençant la congestion
    df = identifier_facteurs_congestion(df)
    
    # Préparation pour la modélisation
    df_modelisation = preparer_pour_modelisation(df)
    
    # Sauvegarder les données préparées
    df.to_csv('donnees_preparees.csv', index=False)
    df_modelisation.to_csv('donnees_modelisation.csv', index=False)
    
    print("\nPhase 1 terminée avec succès!")
    print("Les données préparées ont été sauvegardées dans 'donnees_preparees.csv'")
    print("Les données pour la modélisation ont été sauvegardées dans 'donnees_modelisation.csv'")
    print("Les facteurs influençant la congestion ont été identifiés et sauvegardés dans 'facteurs_influencant_congestion.csv'")
    
    return df, df_modelisation

# Exécution du programme
if __name__ == "__main__":
    # Remplacer par le chemin de votre fichier CSV
    chemin_fichier = "donnees_transport_gare.csv"
    df, df_modelisation = executer_phase1(chemin_fichier)