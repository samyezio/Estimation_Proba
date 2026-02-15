import numpy as np #sert la simulation des lois 
import math 
import matplotlib.pyplot as plt # tracer les graphiques
from loi_poisson import poisson_theorique


def binomiale_theorique(p:float, k:int,n:int) -> float:
 """Calcule la probabilité théorique d'une loi binomiale.

    Args:
        p (float): La probabilité de succès.
        k (int): La valeur pour laquelle on calcule la probabilité.
        n (int): Le nombre d'essais."""
        
 return math.comb(n,k) * (p ** k) * ((1-p) ** (n-k))

def binomiale_simulation(ps,n,nb_tirages):
    """Simule des échantillons suivant une loi de binomiale et trace les histogrammes.
    
        Args:
            lambdas (list): Liste des proba de succes pour les différentes simulations.
            n : le nombre d'essais
            nb_tirages (int): Taille de chaque échantillon simulé."""
    plt.figure(figsize=(15,6))#cree une figure de taille 15x6
    for i, p in enumerate(ps):
        values = np.random.binomial(n,p,nb_tirages)
        # Tracer l'Histogramme la distribution simulee + la distribution theorique P(X=k)
        plt.subplot(2,3,i+1)
        plt.title(f'Simulation de la loi de Binomiale pour p={p}')
        plt.xlabel('k')
        plt.ylabel('Probabilité')             #normalise l'histogramme chaque barre ≈ P(X=k) simulé
        plt.hist(values, bins=range(n+2), density=True , align='left', 
                 color='skyblue', edgecolor='black', alpha=0.6, label=f'Simulation p={p}')
        #fonction de masse theorique P(X=k)
        x_theorique = list(range(n +1)) #P(X=k) théorique
        y_theorique = [binomiale_theorique(p,k,n) for k in x_theorique]
        plt.plot(x_theorique, y_theorique, 'ro-', linewidth=2, markersize=5, label='Théorique P(X=k)')
        plt.legend()
        #fonction de repartition simule 
        plt.subplot(2,3,i+4)
        plt.title(f'fonction de repartition de la loi de binomial pour p={p}')
        plt.xlabel('k')
        plt.ylabel('Probabilité')
        plt.hist(values, bins=range(n+2), density=True, cumulative=True, align='left', histtype='step',
                 color='lightgreen', edgecolor='black', alpha=0.6, label=f'fonction de repartition pour p={p}')
        #fonction de repartition theorique
        y_repart_theo = np.cumsum(y_theorique)
        plt.step(x_theorique, y_repart_theo, where='post', linestyle='--', label='Répartition Théorique')
        plt.legend()
        
    plt.tight_layout()
    plt.suptitle(f"Loi de Binomiale ({nb_tirages} tirages)", y=1.02)
    plt.show()
    
def comp_poisson_binomiale():
    plt.figure(figsize=(15,6))#cree une figure de taille 15x6
    n=100
    paires =[(0.5,50),(0.8,80),(0.05,5)]
    
    for i, (p,l) in enumerate(paires):
        x_theorique = list(range(n +1)) #P(X=k) théorique
        y_theorique=[binomiale_theorique(p,k,n) for k in x_theorique]
        y_theorique_poisson=[poisson_theorique(l,k) for k in x_theorique]
        plt.subplot(1,3,i+1)
        plt.title(f'Comparaison Binomiale (p={p},n={n}) et Poisson (λ={l})')
        plt.xlabel('k')
        plt.ylabel('Probabilité')
        plt.plot(x_theorique, y_theorique, 'bo-', linewidth=2, markersize=5, label='Binomiale P(X=k)')
        plt.plot(x_theorique, y_theorique_poisson, 'ro--', linewidth=2, markersize=5, label='Poisson P(X=k)')
        plt.legend()    
        plt.suptitle("Convergence de la Binomiale vers Poisson (Loi des événements rares)")
    #tu observes que la Binomiale converge vers Poisson quand p est petit et n grand

    #C’est exactement la loi des événements rares
    #C’est exactement la loi des événements rares
    plt.tight_layout()
    plt.show()
    """p=0.5, λ=50 → courbes différentes

Binomiale centrée autour de np=50, mais plus symétrique

Poisson approximative, mais pas exacte (p pas petit)

p=0.8, λ=80 → même chose, pas proche

p=0.05, λ=5 → courbes quasi identiques

Cas « événements rares » : Poisson ≈ Binomiale"""
   
    
    
binomiale_simulation(ps=(0.5,0.8,0.05),n=50,nb_tirages=100)  
print("Affichage de la comparaison Binomiale vs Poisson...")
comp_poisson_binomiale()
        

      
        
        
        
        
        
 
 
        
    

