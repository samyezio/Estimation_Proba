import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk

X_sample = np.random.rand(1000)
noises = np.random.rand(1000) * 0.1 
Y_sample = X_sample + noises 
X_glaces = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])

#sklearn requires 2D arrays for features
X_sample_2D = X_sample.reshape(-1,1)
X_glaces_2D = X_glaces.reshape(-1,1)

model = sk.LinearRegression()

model.fit(X_sample_2D, Y_sample) #entraînement du modèle

Y_pred=model.predict(X_glaces_2D) #prédiction des valeurs Y pour les X_glaces

print("Prédictions pour les glaces :", Y_pred)

plt.figure(figsize=(8,6))
plt.scatter(X_sample,Y_sample,color='blue',label='Donnes entrainement')
plt.scatter(X_glaces,Y_pred, color='red', label='Prédictions pour les glaces'  )
plt.plot(X_glaces, Y_pred, color='red', linestyle='--', label='Modèle linéaire')

plt.title("Régression Linéaire Sklearn : Entraînement vs Prédiction")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

""""Analyse par les graphiquesLe graphique produit est très intéressant
pour l'analyse car il montre deux choses :Le nuage bleu : Représente tes données "théoriques" (entre 0 et 1).
La ligne et les points rouges : Montrent comment le modèle projette sa connaissance très loin (entre 11 et 25).
Note d'analyse : Tu remarqueras que les prédictions rouges sont parfaitement alignées. 
C'est normal, car predict ne fait qu'appliquer la formule mathématique $y = ax + b$
sans ajouter de bruit."""