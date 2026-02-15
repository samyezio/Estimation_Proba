# Proba-Stat — Projets Python

Dépôt regroupant mes projets et TP en **probabilités / statistiques**, réalisés en **Python** (simulation de lois, estimation, intervalles de confiance, régression et évaluation).

## Objectifs
- Simuler et visualiser des **lois de probabilité**
- Comparer **théorie vs simulation** (histogrammes, densités, fonctions de répartition)
- Appliquer des notions de **statistique inférentielle** : estimation, IC (Normal/Student), proportions
- Mettre en pratique des méthodes de **régression linéaire** et évaluer la qualité des prédictions

## Contenu du dépôt

### 1) Simulations de lois
- **Loi exponentielle** : densité + fonction de répartition, comparaison simulée/théorique  
  Fichier : `loi_expo.py` :contentReference[oaicite:0]{index=0}
- **Loi normale** : densité + répartition (approx par somme), + illustration TCL via moyennes d’une uniforme  
  Fichier : `loi_normale.py` :contentReference[oaicite:1]{index=1}
- **Loi de Poisson** : PMF + CDF, comparaison simulée/théorique  
  Fichier : `loi_poisson.py` :contentReference[oaicite:2]{index=2}
- **Loi binomiale** : PMF + CDF, + comparaison Binomiale vs Poisson (événements rares)  
  Fichier : `loi_binomiale.py` :contentReference[oaicite:3]{index=3}

### 2) Estimation & intervalles de confiance
- Moyenne empirique + histogramme (temps de réaction)
- IC de la moyenne :
  - variance connue (loi normale)
  - variance inconnue (loi de Student)
- IC d’une proportion (approx normale)
Fichier : `Estimateur.py` :contentReference[oaicite:4]{index=4}  
Image exemple : `histogramme_reactions.png`

### 3) Régression linéaire
- Régression linéaire avec **scikit-learn** (entraînement + prédictions + visualisation)  
  Fichier : `sklearm.py` :contentReference[oaicite:5]{index=5}
- Régression par **moindres carrés** (calcul analytique de a, b + SSE)  
  Fichier : `meth_moindres_carres.py` :contentReference[oaicite:6]{index=6}
- Régression par **descente de gradient** (avec normalisation + critère d’arrêt)  
  Fichier : `meth_gradiant.py` :contentReference[oaicite:7]{index=7}

### 4) Qualité des prédictions
- **RMSE**
- **R²** (coefficient de détermination)
Fichier : `qualite_des_predictions.py` :contentReference[oaicite:8]{index=8}

## Prérequis
- Python 3.10+ (recommandé)
- Bibliothèques :
  - `numpy`
  - `matplotlib`
  - `scipy` (pour Student / quantiles dans certains scripts) :contentReference[oaicite:9]{index=9}
  - `scikit-learn` (pour la partie régression sklearn) :contentReference[oaicite:10]{index=10}

Installation rapide :

pip install numpy matplotlib scipy scikit-learn

## Exécution

Lancer un script :

python loi_poisson.py
python loi_normale.py
python Estimateur.py

## Notes

-Plusieurs scripts affichent des graphiques (histogrammes, densités, répartition, droites de régression).

-Certains scripts impriment aussi des résultats numériques (IC, RMSE, R², etc.).
