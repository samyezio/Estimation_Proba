import numpy as np
import matplotlib.pyplot as plt
import math

def exp_theorique(lam: float, x: int) -> float:
    return lam * np.exp(-lam * x)  #fonction de densité f(x)=λe^(−λx)

def exp_repartition_theorique( x: float,lam : float) -> float:
    return 1 - np.exp(-lam * x)  #fonction de répartition F(x)=1−e^(−λx)

def plot_exponentielle(lambdas, nb_tirages):
    plt.figure(figsize=(12, 8))

    for i, lam in enumerate(lambdas):
        #pour numpy faut mettre l'inverse c pas lamda mais le scale du coup
        values = np.random.exponential(scale=1/lam, size=nb_tirages)
        
        plt.subplot(2, 2, i + 1)
        plt.title(f"Densité $\lambda={lam}$)")
        
        plt.hist(values, density=True, bins=20, color='lightgreen', 
                 edgecolor='black', alpha=0.6, label='Simulée')
        x_theo = np.linspace(0, max(values), 200)
        y_pdf = [exp_theorique(lam, x) for x in x_theo] # fonction de densité théorique

        plt.plot(x_theo, y_pdf, 'r-', linewidth=2, label='Théorique')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(2, 2, i + 3) 
        plt.title(f"Répartition  $\lambda={lam}$")
        
        plt.hist(values, density=True, cumulative=True, histtype='step',
                 bins=100, linewidth=2, color='green', label='Simulée')
        
        y_rep = [exp_repartition_theorique(y,lam) for y in y_pdf]
        plt.plot(x_theo, y_rep, 'r--', label='Théorique')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

    plt.suptitle(f"Loi Exponentielle : {nb_tirages} tirages", y=1.02)
    plt.tight_layout()
    plt.show()

plot_exponentielle(lambdas=[0.5, 2], nb_tirages=100)