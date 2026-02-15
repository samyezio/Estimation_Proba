import numpy as np
import math
import matplotlib.pyplot as plt 

def normal_theorique(x:float,moy:float,ecart_type:float)-> float:
    """Calcule la densité théorique d'une loi normale.

    Args:
        x (float): La valeur pour laquelle on calcule la densité.
        moy (float): La moyenne de la loi normale.
        ecart_type (float): L'écart type de la loi normale."""
    return (1 / ecart_type * math.sqrt(2 * math.pi)) * math.exp(-((x-moy)** 2 / ( 2 * ecart_type ** 2)))

def normale_simulation(moyenne,ecart_type,nb_tirages):
    """Simule des échantillons suivant une loi normale et trace les histogrammes.
    
        Args:
            moyenne (float): La moyenne de la loi normale.
            ecart_type (float): L'écart type de la loi normale.
            nb_tirages (int): Taille de chaque échantillon simulé."""
    plt.figure(figsize=(15,6))
    values=np.random.normal(moyenne,ecart_type,nb_tirages)
    # Tracer l'Histogramme la distribution simulee 
    plt.subplot(1,2,1)
    plt.title(f'Simulation de la loi Normale pour μ={moyenne}, σ={ecart_type}')
    plt.xlabel('x')
    plt.ylabel('Densité de probabilité')
    plt.hist(values, bins=30, density=True , 
             color='skyblue', edgecolor='black', alpha=0.6, label=f'Simulation μ={moyenne}, σ={ecart_type}')
    #fonction de densité theorique f(x)
    x_theorique = np.linspace(min(values),max(values),200) # on cree 200 points entre minimum et maximum des valeurs des points simule qui suit une loi normale
    y_theorique=[normal_theorique(x,moyenne,ecart_type) for x in x_theorique]
    plt.plot(x_theorique,y_theorique,'r-',linewidth=2,markersize=5,label='Théorique f(x)')
    plt.legend()
    #fonction de repartition simule
    plt.subplot(1,2,2)
    plt.title(f'focntion de reparition simule de la loi normale pour  μ={moyenne} , σ={ecart_type}') 
    plt.xlabel('x')
    plt.ylabel('probabilite')
    plt.hist(values,bins=30,cumulative=True,density=True,histtype='step',
             color='lightgreen',edgecolor='black',alpha=0.6,label=f'fonction de repartition pour μ={moyenne}, σ={ecart_type}')
    #fonction de repartition theorique
    dx = x_theorique[1] - x_theorique[0] # pas entre deux points consecutifs F(x)≈∑f(xi​)Δx hauteur = f(x) largeur = dx , on a y_theorique est la densite f(x) donc F(x)=∫−∞x​f(t)dt
    plt.plot(x_theorique, np.cumsum(y_theorique) * dx, 'r--', label="Répartition théorique")
    plt.legend()
    plt.tight_layout()
    plt.suptitle(f"Loi Normale ({nb_tirages} tirages)", y=1.02)
    plt.show()
    
def loi_uniforme(n:int,echantillon:int):
    """Simule des échantillons suivant une loi uniforme et trace les histogrammes.
    
        Args:
            n (int): La borne supérieure de la loi uniforme [0,n).
            nb_tirages (int): Taille de chaque échantillon simulé."""
    plt.figure(figsize=(15,6))
    values=np.random.uniform(0,1,(echantillon,n)) #10000 échantillons de n (n = 30) variables suivant une loi uniforme sur [0;1]
    moyennes = np.mean(values , axis=0)# calcul des moyennes des échantillons par colonne car axis=0
    plt.hist(moyennes, bins=30, density=True , 
             color='skyblue', edgecolor='black', alpha=0.6, label=f'Simulation des moyennes de {echantillon} échantillons de taille {n}'   )
    #fonction de densité theorique f(x)
    moyenne=0.5
    ecart_type = 1 / math.sqrt(12 * n)
    x_theorique = np.linspace(min(moyennes),max(moyennes),200) # on cree 200 points entre minimum et maximum des valeurs des points simule qui suit une loi normale
    y_theorique=[normal_theorique(x,moyenne,ecart_type) for x in x_theorique]
    plt.plot(x_theorique,y_theorique,'r-',linewidth=2,markersize=5,label='Théorique f(x)')
    plt.legend()
    plt.show()
    
    
normale_simulation(moyenne=20,ecart_type=math.sqrt(12.5),nb_tirages=1000)
loi_uniforme(n=30,echantillon=10000)
#c   
#Bien que les variables initiales suivent une loi uniforme,
# la distribution des moyennes des échantillons présente une forme proche 
# d’une loi normale lorsque la taille de l’échantillon est suffisamment grande.  

#Le théorème central limite (TCL) car
#La somme (ou la moyenne) de variables aléatoires indépendantes 
# et identiquement distribuées, de variance finie, converge en loi 
# vers une loi normale lorsque la taille de l’échantillon tend vers l’infini.

#Plus la taille de l’échantillon n augmente, 
#plus l’approximation par la loi normale est précise.