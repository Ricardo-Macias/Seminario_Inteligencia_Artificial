import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")

import matplotlib.pyplot as plt
import numpy as np
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Contour
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Surf
from IPython import display

def Seleccion(aptitud):
    Idx = np.argsort(aptitud)
    Idx = Idx[::-1]
    N = (aptitud.size)

    rank= np.arange(N,-1,-1)
    rank_total = np.sum(rank)

    r = np.random.rand()
    p_sum = 0

    for i in range(N):
        p_sum = p_sum + rank[i] / rank_total

        if p_sum >= r:
            n = Idx[i]
            return n
        
        n = Idx[N-1]
        return n

# --------------------
# FUNCIONES OBJETIVOS
# --------------------

Griewank = lambda x,y: ((x**2/4000)+(y**2/4000))-(np.cos(x)* np.cos(y/np.sqrt(2))) + 1
Rastrigin = lambda x,y: 10*2 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
Sphere = lambda x,y: x**2 + y**2

# --------------------
# PARAMETROS
# --------------------

xl = np.array([-5,-5])
xu = np.array([5,5])

D = 2
G = 150

N = 50
L = 15

pf = 30
po = N - pf

x = np.zeros((D,pf))
l = np.zeros(pf)
aptitud = np.zeros(pf)
fitness = np.zeros(pf)

for i in range (pf):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = Sphere(x[0,i], x[1,i])

fx_plot = np.zeros(G)

# --------------------
# ABC
# --------------------

for g in range(G):

    for i in range(pf):
        k = i
        while k == i:
            k = np.random.randint(pf)
        
        j = np.random.randint(D)
        phi = 2 * np.random.rand() - 1

        v = x[:,i].copy()
        v[j] = x[j, i] + phi * (x[j, i] - x[j, k])

        fv = Sphere(v[0], v[1])

        if fv < fitness[i]:
            x[:,i] = v
            fitness[i] = fv
            l[i] = 0
        else:
            l[i] = l[i] + 1
        
        if fitness[i] >= 0:
            aptitud[i] = 1/(1+ fitness[i])
        else:
            aptitud[i] = 1 + np.abs(fitness[i])

    
    for i in range(po):
        m = Seleccion(aptitud)

        k = m
        while k == m:
            k = np.random.randint(pf)
        
        j = np.random.randint(D)
        phi = 2 * np.random.rand() - 1

        v = x[:,m].copy()
        v[j] = x[j,m] + phi * (x[j,m] - x[j,k])

        fv = Sphere(v[0],v[1])

        if fv < fitness[m]:
            x[:,m] = v
            fitness[m] = fv

            l[m] = 0
        else:
            l[m] = l[m] + 1
    
    for i in range(pf):
        if l[i] > L:
            x[:,i] = xl + (xu - xl) * np.random.rand(D)
            fitness[i] = Sphere(x[0,i], x[1, i])
            l[i] = 0
    
    fx_plot[g] = np.min(fitness)

igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",
      x[1, igb], " f(x,y)=", fitness[igb])
Plot_Contour.plot_contour(Sphere, x, xl, xu)
Plot_Surf.plot_surf(Sphere,x,xl,xu,igb)
plt.plot(fx_plot)
plt.title("Convergencia")
plt.draw()
plt.show()
