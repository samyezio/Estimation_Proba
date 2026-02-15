import numpy as np
import matplotlib.pyplot as plt 


def moindres_carres(X,Y):
    """
    Calcule les coefficients a et b de la droite y = a*x + b qui minimise la somme des carrés des écarts entre les points (X[i], Y[i]) et la droite.
    
    Arguments:
    X -- liste ou tableau numpy des abscisses des points
    Y -- liste ou tableau numpy des ordonnées des points
    
    Retourne:
    a -- coefficient directeur de la droite
    b -- ordonnée à l'origine de la droite
    # Estimation des coefficients de régression linéaire simple :
# â = ( n * Σ(x_i * y_i) - Σ(x_i) * Σ(y_i) ) / ( n * Σ(x_i^2) - (Σ(x_i))^2 )
# b̂ = ( Σ(y_i) / n ) - â * ( Σ(x_i) / n )
# où :
#   x_i, y_i = données observées
#   n        = nombre d'observations
#   â       = pente de la droite de régression
#   b̂       = ordonnée à l'origine (intercept)
    Le but est de trouver a et b tels que la somme des carrés des écarts entre les points (X[i], Y[i]) et la droite y = a*x + b soit minimale.
    min S(a,b) minimise l'erreur totale S(a,b) = Σ (Y[i] - (a*X[i] + b))^2
    """
    x_sum=np.sum(X)
    y_sum=np.sum(Y)
    xx_sum=np.sum(X*X)
    xy_sum=np.sum(X*Y)
    n=len(X)
    a = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum ** 2)
    b=(y_sum - a * x_sum)/n
    Y_pred=a*X +b
    S=np.sum((Y- Y_pred) ** 2)
    return a,b,S


X_glaces = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])
Y_glaces = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408])

a_glaces,b_glaces,S_glaces = moindres_carres(X_glaces,Y_glaces)
plt.figure(figsize=(8,6))
plt.scatter(X_glaces, Y_glaces, color='blue', label='Données')
plt.plot(X_glaces, a_glaces * X_glaces + b_glaces, color='red', label='Droite de régression')
plt.xlabel("Température (°C)")
plt.ylabel("Ventes (€)")
plt.legend()
plt.show()

X_sample = np.random.rand(1000)
noises = np.random.rand(1000) * 0.1
Y_sample = X_sample + noises

a_sample,b_sample,S_sample = moindres_carres(X_sample,Y_sample)
plt.figure(figsize=(8,6))
plt.scatter(X_sample, Y_sample, color='blue', label='Données échantillon')
plt.plot(X_sample, a_sample * X_sample + b_sample, color='red', label='Droite de régression')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()  

#Pour les glaces : On trouve $a \approx 30.09$.
# Cela signifie que pour chaque degré Celsius supplémentaire,
# les ventes augmentent en moyenne de $30,09$ €.

#Pour les données synthétiques : On remarque que $a$ est très proche de $1$ (notre $\alpha$). 
# La valeur de $b$ est proche de $0.05$ car le bruit ajouté (entre $0$ et $0.1$) a une moyenne de $0.05$,
# ce qui "rehausse" légèrement la droite.