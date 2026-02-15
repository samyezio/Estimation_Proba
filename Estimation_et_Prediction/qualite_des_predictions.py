import numpy as np
import matplotlib.pyplot as plt
import math 

def RMSE(Y_true , Y_pred):
    """
    Calcule la racine de l'erreur quadratique moyenne (RMSE) entre les valeurs réelles et les valeurs prédites.
    
    Arguments:
    Y_true -- liste ou tableau numpy des valeurs réelles
    Y_pred -- liste ou tableau numpy des valeurs prédites
    
    Retourne:
    rmse -- la valeur de la RMSE
    """
    n = len(Y_true)
    rmse=math.sqrt((np.sum((Y_pred - Y_true) ** 2)) / n)
    return rmse

# Exemple d'utilisation de la fonction RMSE
Y_true = np.array([3.0, -0.5, 2.0, 7.0])
Y_pred = np.array([2.5, 0.0, 2.0, 8.0])     
rmse_value = RMSE(Y_true, Y_pred)
print("Valeur de la RMSE :", rmse_value)


def coefficient_determination(Y_true, Y_pred):
    """
    Calcule le coefficient de détermination R² entre les valeurs réelles et les valeurs prédites.
    
    Arguments:
    Y_true -- liste ou tableau numpy des valeurs réelles
    Y_pred -- liste ou tableau numpy des valeurs prédites
    
    Retourne:
    r2 -- la valeur du coefficient de détermination R²
    """
    num = np.sum ((Y_pred - np.mean(Y_true)) **2)
    denum = np.sum ((Y_true - np.mean(Y_true)) ** 2)
    r2 = num / denum
    return r2

# Imaginons que tu as déjà calculé Y_pred avec sklearn ou tes fonctions précédentes
# sur les données des glaces réelles :
X_glaces = np.array([14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2])
Y_glaces_reelles = np.array([215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408])
# Calculons les prédictions avec les coefficients a et b trouvés en question 2
a, b = 30.0879, -159.4742
Y_glaces_pred = a * X_glaces + b

# Calcul de la qualité
rmse_glaces = RMSE(Y_glaces_reelles, Y_glaces_pred)
r2_glaces = coefficient_determination(Y_glaces_reelles, Y_glaces_pred)

print(f"Qualité du modèle Glaces :")
print(f"RMSE : {rmse_glaces:.2f} €")
print(f"R²   : {r2_glaces:.4f}")
""""Si $R^2$ est proche de 1, le modèle explique très bien les données.
Si $R^2$ est proche de 0, le modèle n'est pas meilleur qu'une simple moyenne."""
    