# 🚚 Prédiction du Temps de Livraison

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 📋 Description

Projet de Machine Learning pour prédire le temps total de livraison d'une commande, de la préparation jusqu'à la réception par le client. Ce modèle aide les entreprises de logistique à anticiper les retards, optimiser les tournées et améliorer l'expérience client.

## 🎯 Objectifs

- **Anticiper les retards** de livraison
- **Informer les clients** avec des estimations précises en temps réel
- **Optimiser l'organisation** des tournées de livraison
- Développer un modèle **automatisé, testé et intégrable en production**

## 📊 Dataset

Le modèle prédit la variable cible `DeliveryTime` en utilisant les caractéristiques suivantes :

| Variable | Description |
|----------|-------------|
| `Distance_km` | Distance entre le restaurant et l'adresse de livraison |
| `Traffic_Level` | Niveau de trafic routier |
| `Vehicle_Type` | Type de véhicule utilisé |
| `Time_of_Day` | Heure de la journée |
| `Courier_Experience` | Expérience du livreur |
| `Weather` | Conditions météorologiques |
| `Preparation_Time` | Temps de préparation de la commande |

## 🏗️ Structure du Projet

```
prediction_du_Temps_de_Livraison/
│
├── data/
│   └── dataset.csv                 # Données d'entraînement
│
├── functions/
│   └── fonctions.py                # Fonctions de prétraitement et modélisation
│
├── Nootbooks/
│   └── EDA.py                      # Analyse exploratoire des données
│
├── tests/
│   └── test_fonctions.py           # Tests unitaires
│
├── .github/
│   └── workflows/
│       └── python-tests.yml        # CI/CD avec GitHub Actions
│
├── requirements.txt                # Dépendances du projet
└── README.md                       # Documentation
```

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/<ton-utilisateur>/prediction_du_Temps_de_Livraison.git
cd prediction_du_Temps_de_Livraison
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Exécuter l'analyse exploratoire

```bash
python Nootbooks/EDA.py
```

### Lancer les tests unitaires

```bash
pytest -v
```

## 🧪 Méthodologie

### 1. Analyse Exploratoire (EDA)
- Heatmap de corrélation des variables numériques
- Countplots pour la distribution des variables catégorielles
- Boxplots pour analyser les relations avec `DeliveryTime`
- Analyse de la distribution de la variable cible

### 2. Prétraitement
- **StandardScaler** : normalisation des variables numériques
- **OneHotEncoder** : encodage des variables catégorielles
- **SelectKBest** (f_regression) : sélection des features les plus pertinentes

### 3. Modélisation
Deux modèles testés avec **GridSearchCV** :
- **RandomForestRegressor**
- **SVR (Support Vector Regression)**

**Métrique principale** : MAE (Mean Absolute Error)

### 4. Pipeline scikit-learn
Pipeline automatisé intégrant :
- Prétraitement (scaling + encodage)
- Sélection de features
- Modélisation

✅ **Avantage** : Évite les fuites de données et simplifie le déploiement

## 🧪 Tests Automatisés

Les tests unitaires vérifient :
- La cohérence des dimensions des données
- Le format des prédictions
- La performance du modèle (seuil MAE maximum)

```bash
pytest -v tests/
```

## ⚙️ Intégration Continue (CI/CD)

**GitHub Actions** exécute automatiquement les tests à chaque push sur `main`.

Le workflow `.github/workflows/python-tests.yml` :
- Installe Python 3.11
- Installe les dépendances
- Exécute les tests avec pytest
- Affiche une alerte si un test échoue

## 📈 Résultats

| Modèle | MAE | R² |
|--------|-----|-----|
| RandomForestRegressor | 7.608370238095238|  0.7469384552378573|
| SVR | 6.058334766827715 | 0.8149002107491989 |

**Modèle retenu** : 
 -Le SVR (Support Vector Regressor) est le meilleur choix dans ce cas car :

 -Il a une erreur moyenne plus faible (MAE)

 -Il explique mieux la variance des données (R² plus élevé)

## 🛠️ Technologies

- **Python 3.11+**
- **pandas**, **numpy** : manipulation de données
- **matplotlib**, **seaborn** : visualisation
- **scikit-learn** : modélisation et pipelines
- **joblib** : sauvegarde du modèle
- **pytest** : tests unitaires
- **GitHub Actions** : CI/CD

## 📅 Planning

- **Lancement** : 13/10/2025 à 10h00
- **Date limite** : 17/10/2025 à 17h00
- **Durée** : 5 jours

## 🎓 Évaluation

**Durée totale** : 25 minutes

1. **Présentation** (5 min) : Objectif, dataset, approche
2. **Démonstration** (5 min) : Notebook, scripts, résultats
3. **Échanges techniques** (10 min) : Code, modèles, GridSearch, tests
4. **Discussion finale** (5 min) : Organisation, CI/CD, reproductibilité

## 👤 Auteur

Maryem Elbergui

**Linkedin:** https://www.linkedin.com/in/maryem-elbergui-0939401b7



 **N'hésite pas à star le projet si tu le trouves utile !**
