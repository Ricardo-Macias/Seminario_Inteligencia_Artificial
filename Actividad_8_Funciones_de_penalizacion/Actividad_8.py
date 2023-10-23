import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")

import matplotlib.pyplot as plt
import numpy as np
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Contour
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Surf
from IPython import display

# ---------------------------------------
#           PENALIZACION
# ---------------------------------------

def penalty(x, xl, xu):
    D = x.size
    z = 0

    for j in range(D):
        if x[j] < xl[j]:
            z = z + 1
        elif x[j] > xu[j]:
            z = z + 1
        else:
            z = z + 0
    
    #for j in range(D):
    #   if[j] < xl[j]:
    #        z = z + (x[j] - xl[j]) ** 2
    #    elif x[j] > xu[j]:
    #        z = z + (x[j] - xu[j]) ** 
    #    else:
    #        z = z + 0

    return z

# ---------------------------------------
#      FUNCION OBJETIVO 
# ---------------------------------------

f = lambda x, y: np.sin(x+y) + (x-y) ** 2 - 1.5*x + 2.5*y + 1
fp = lambda x, xl, xu: f(x[0], x[1]) + 1000 * penalty(x, xl, xu) 

# ---------------------------------------
#       PARAMETROS
# ---------------------------------------

xl = np.array([-1.5, -3])
xu = np.array([4,4])

G = 30
N = 50
D = 2

w = 0.6
c1 = 2
c2 = 2

x = np.zeros((D,N))
xp = np.zeros((D,N))
v = np.zeros((D,N))
fitness = np.zeros(N)

f_plot = np.zeros(G)

# ---------------------------------------
#           PSO
# ---------------------------------------

for i in range(N):
    x[:,i] = xl + (xu - xl) * np.random.rand(D)
    xp[:,i] = x[:,i]
    v[:,i] = 0.5 * np.random.randn(D)
    fitness[i] = f(x[0, i], x[1, i])

for g in range(G):
    for i in range(N):
        fx = f(x[0,i], x[1,i])

        if fx < fitness[i]:
            xp[:,i] = x[:,i]
            fitness[i] = fx
        
        ig = np.argmin(fitness)
    
    for i in range(N):
        v[:,i] = w*v[:,i] + c1 * np.random.rand() * c2 * np.random.rand() * (xp[:,ig] - x[:,i])
        x[:,i] = x[:,i] + v[:,i]
    
    f_plot[g] = np.min(fitness)

print("Mínimo global en x=", xp[0, ig], " y=", xp[1, ig], " f(x,y)=", f(
    xp[0, ig], xp[1, ig]))  # --------------> Funcion Objetivo
Plot_Contour.plot_contour(f, x, xl, xu)  # --------------> Funcion Objetivo
Plot_Surf.plot_surf(f, x, xl, xu, ig)  # --------------> Funcion Objetivo
plt.plot(range(G), f_plot)
plt.title("Convergencia")
plt.draw()
plt.show()
