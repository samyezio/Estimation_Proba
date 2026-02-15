import numpy as np #sert la simulation des lois 
import math 
import matplotlib.pyplot as plt # tracer les graphiques

def poisson_theorique(lamda:float, k:int) -> float:
 """Calcule la probabilité théorique d'une loi de Poisson.

    Args:
        lambda (float): Le paramètre lambda de la loi de Poisson.
        k (int): La valeur pour laquelle on calcule la probabilité."""
        
 return math.exp(-lamda) * (lamda ** k)/math.factorial(k)

def poison_simulation(lambdas,n):
    """Simule des échantillons suivant une loi de Poisson et trace les histogrammes.
    
        Args:
            lambdas (list): Liste des paramètres lambda pour les différentes simulations.
            n (int): Taille de chaque échantillon simulé."""
    plt.figure(figsize=(15,6))#cree une figure de taille 15x6
    for i, lamda in enumerate(lambdas):
        values = np.random.poisson(lamda,n)
        k_max = max(values) + 2
        # Tracer l'Histogramme la distribution simulee + la distribution theorique P(X=k)
        plt.subplot(2,3,i+1)
        plt.title(f'Simulation de la loi de Poisson pour λ={lamda}')
        plt.xlabel('k')
        plt.ylabel('Probabilité')             #normalise l'histogramme chaque barre ≈ P(X=k) simulé
        plt.hist(values, bins=range(k_max+2), density=True , align='left', 
                 color='skyblue', edgecolor='black', alpha=0.6, label=f'Simulation λ={lamda}')
        #fonction de masse theorique P(X=k)
        x_theorique = list(range(k_max +1)) #P(X=k) théorique
        y_theorique = [poisson_theorique(lamda,k) for k in x_theorique]
        plt.plot(x_theorique, y_theorique, 'ro-', linewidth=2, markersize=5, label='Théorique P(X=k)')
        plt.legend()
        #fonction de repartition simule 
        plt.subplot(2,3,i+4)
        plt.title(f'fonction de repartition Simulation de la loi de Poisson pour λ={lamda}')
        plt.xlabel('k')
        plt.ylabel('Probabilité')
        plt.hist(values, bins=range(k_max+2), density=True, cumulative=True, align='left', histtype='step',
                 color='lightgreen', edgecolor='black', alpha=0.6, label=f'fonction de repartition pour λ={lamda}')
        #fonction de repartition theorique
        y_repart_theo = np.cumsum(y_theorique)
        plt.step(x_theorique, y_repart_theo, where='post', linestyle='--', label='Répartition Théorique')
        plt.legend()
        
    plt.tight_layout()
    plt.suptitle(f"Loi de Poisson ({n} tirages)", y=1.02)
    plt.show()
    
poison_simulation(lambdas=(1,10,30),n=100)  #si n=1000 on voit mieux la convergence vers la loi theorique des grandes nombres      
        

      
        
        
        
        
        
 
 
        
    

