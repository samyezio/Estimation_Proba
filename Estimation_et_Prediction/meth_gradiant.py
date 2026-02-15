import numpy as np
import matplotlib.pyplot as plt 

def meth_gradiant(X,Y,learning_rate=0.5):
    """
    Calcule les coefficients a et b de la droite y = a*x + b en utilisant la méthode du gradient pour minimiser la somme des carrés des écarts entre les points (X[i], Y[i]) et la droite.
    
    Arguments:
    X -- liste ou tableau numpy des abscisses des points
    Y -- liste ou tableau numpy des ordonnées des points
    learning_rate -- taux d'apprentissage pour la mise à jour des coefficients
    iterations -- nombre d'itérations pour l'algorithme de gradient
    
    Retourne:
    a -- coefficient directeur de la droite
    b -- ordonnée à l'origine de la droite
    """
    n=len(X)
    a=0.0
    b=0.0
    Y_pred=a * X + b
    J_a_b_prev =( 1 / 2 * n ) * np.sum((Y_pred - Y) ** 2)
    
    while(True):
        D_a = (1 / n ) * np.sum(X * (Y_pred - Y))
        D_b = (1 / n ) * np.sum(Y_pred - Y)
        a = a - learning_rate * D_a
        b = b - learning_rate * D_b
        Y_pred=a * X + b
        J_a_b_curr =( 1 / 2 * n ) * np.sum((Y_pred - Y) ** 2)
        if abs(J_a_b_curr - J_a_b_prev) < 10 ** - 6:
            break
        J_a_b_prev=J_a_b_curr
    
    return a,b
    
X_glaces = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])
Y_glaces = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408])    
x_min,x_max = X_glaces.min(), X_glaces.max()
y_min,y_max = Y_glaces.min(), Y_glaces.max()

X_norm = (X_glaces - x_min) / (x_max - x_min)
Y_norm = (Y_glaces - y_min) / (y_max - y_min)

a_glaces,b_glaces = meth_gradiant(X_norm,Y_norm,learning_rate=0.5)

plt.figure(figsize=(8,6))
plt.scatter(X_norm, Y_norm, color='blue', label='Données normalisées')
plt.plot(X_norm, a_glaces * X_norm + b_glaces, color='red', label='Droite de régression (Moindres carrés)')
plt.xlabel("Température (°C)")
plt.ylabel("Ventes (€)")
plt.legend()
plt.show()

X_sample = np.random.rand(1000)
noises = np.random.rand(1000) * 0.1
Y_sample = X_sample + noises

a_sample,b_sample = meth_gradiant(X_sample,Y_sample,learning_rate=0.5)
plt.figure(figsize=(8,6))
plt.scatter(X_sample, Y_sample, color='blue', label='Données échantillon')
plt.plot(X_sample, a_sample * X_sample + b_sample, color='red', label='Droite de régression')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()     
    
   