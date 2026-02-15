import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- DONNÉES 2.1 ---
reactions = np.array([
    0.98, 1.4, 0.84, 0.86, 0.54, 0.68, 1.35, 0.76, 0.79, 0.99,
    0.88, 0.75, 0.45, 1.09, 0.68, 0.60, 1.13, 1.30, 1.20, 0.91,
    0.74, 1.03, 0.61, 0.98, 0.91
])
n = len(reactions)

# 1(a) Moyenne empirique
moyenne = np.mean(reactions)
print(f"2.1 - Moyenne empirique : {moyenne:.3f}")

# 1(b) Histogramme
plt.hist(reactions, bins=8, color='skyblue', edgecolor='black')
plt.title('Histogramme des temps de réaction (n=25)')
plt.xlabel('Temps (s)')
plt.ylabel('Nombre de conducteurs')
plt.savefig('histogramme_reactions.png')

# Intervalle de confiance (IC) pour la moyenne :
# IC = [ X̄_n - t_(α/2) * s / √n , X̄_n + t_(α/2) * s / √n ]
# où :
#   X̄_n = moyenne de l'échantillon
#   s   = écart-type de l'échantillon
#   n   = taille de l'échantillon
#   t_(α/2) = quantile de la loi de Student pour un niveau de confiance 1-α

#on a 2 cas 
# où t_(α/2) est défini en fonction de α et s est l'écart-type de l'échantillon :
# - Si la variance σ de la loi est connue :
#     s = σ
#     t_(α/2) = quantile de la loi normale N(0,1) d'ordre 1-α
# - Si la variance est inconnue :
#     s = S_n (écart-type empirique)
#     t_(α/2) = quantile de la loi de Student St(n-1) d'ordre 1-α

#Pourquoi utiliser Student ?Quand $n$ est petit (moins de 30 mesures),
# l'estimation de l'écart-type est moins fiable. La loi de Student "pénalise" cette petite taille d'échantillon 
# en donnant un intervalle de confiance plus vaste, ce qui est mathématiquement plus honnête.


# 1(c) IC - Variance connue sigma^2 = 0.25 -> sigma = 0.5
sigma = 0.5

def ic_normale(confiance): # IC pour la moyenne avec variance connue
    
    alpha = 1 - confiance #On calcule le risque d'erreur $\alpha$. Si la confiance est de 95% (0,95), le risque $\alpha$ est de 5% (0,05). 
    #C'est la probabilité que la vraie moyenne $\mu$ soit en dehors de l'intervalle. car 1 - alpha = confiance
    
    # stats.norm.ppf donne le fractile pour la loi normale
    t = stats.norm.ppf(1 - alpha/2) #fractile t alpha/2 pour la loi normale N(0,1)
    #ppf signifie Percent Point Function.
    # Comme l'intervalle est centré, on partage le risque $\alpha$ en deux :
    # $\alpha/2$ à gauche et $\alpha/2$ à droite.
    # On cherche donc la valeur qui laisse $1 - \alpha/2$ de probabilité à sa gauche.
    
    marge = t * (sigma / np.sqrt(n))
    return (moyenne - marge, moyenne + marge)

print(f"IC 95% (V. connue) : {ic_normale(0.95)}")
print(f"IC 99% (V. connue) : {ic_normale(0.99)}")

# 2. IC - Variance inconnue (Loi de Student)
s_n = np.std(reactions, ddof=1) # Ecart-type empirique (avec n-1 degrés de liberté) Sn
def ic_student(confiance):
    alpha = 1 - confiance
    # stats.t.ppf pour la loi de Student avec n-1 degrés de liberté
    t = stats.t.ppf(1 - alpha/2, df=n-1) #on cherche la fractile t alpha/2 pour la loi de Student avec n-1 degrés de liberté
    marge = t * (s_n / np.sqrt(n))
    return (moyenne - marge, moyenne + marge)

print(f"IC 95% (V. inconnue) : {ic_student(0.95)}")
print(f"IC 99% (V. inconnue) : {ic_student(0.99)}")


# Intervalle de confiance (IC) pour une proportion p d'une loi binomiale B(p) à 1-α % :
# IC = [ X_n - z_(α/2) * sqrt(X_n * (1 - X_n) / n), X_n + z_(α/2) * sqrt(X_n * (1 - X_n) / n) ]
# où :
#   X_n       = proportion observée dans l'échantillon (X_n = nombre de succès / n)
#   n         = taille de l'échantillon
#   z_(α/2)   = fractile supérieur d'ordre α/2 de la loi normale N(0,1)



# --- DONNÉES 2.2 (Proportion) ---
n_prop = 1000
succes = 673
X_n = succes / n_prop
confiance_prop = 0.80
alpha_prop = 1 - confiance_prop

# Fractile z pour 1 - 0.20/2 = 0.90
z_prop = stats.norm.ppf(1 - alpha_prop/2)
marge_prop = z_prop * np.sqrt((X_n * (1 - X_n)) / n_prop)
ic_prop = (X_n - marge_prop, X_n + marge_prop)

print(f"\n2.2 - Proportion estimée : {X_n}")
print(f"IC 80% de la proportion : [{ic_prop[0]:.4f}, {ic_prop[1]:.4f}]")