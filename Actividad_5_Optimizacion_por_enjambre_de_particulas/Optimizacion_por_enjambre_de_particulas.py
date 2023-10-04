import matplotlib.pyplot as plt
import numpy as np
from Plot_Contour import *
from Plot_Surf import *
from IPython import display

#FUNCIONES OBJETIVO
Griewank = lambda x,y: ((x**2/4000)+(y**2/4000))-(np.cos(x) * np.cos(y/np.sort(2))) + 1
Rastrigin = lambda x,y: 10**2 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)
Sphere = lambda x,y: x**2 + y**2

xl = np.array([-5,-5])
xu = np.array([5,5])

G = 20
N = 50
D = 2

w = 0.6
c1 = 2
c2 = 2

#IWAPSO
w_max = 0.8
w_min = 0.1

x = np.zeros((D,N))
xp = np.zeros((D,N))
v = np.zeros((D,N))
fitness = np.zeros(G)

f_plot = np.zeros(G)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    xp[:,i] = x[:,i]
    v[:,i] = 0.5 * np.random.randn(D)
    fitness[i] = #Funcion

#PSO

for g in range(G):
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plot_contour("Funcion Objetivo",x,xl,xu)

    for i in range(N):
        fx = #Funcion Objetivo

        if fx < fitness[i]:
            xp[:,i] = x[:, i]
            fitness[i] = fx
        
        ig = np.argmin(fitness)
    
    for i in range(N):
        v[:,i] = w * v[:, i] + c1 * np.random.rand() * (xp[:,i] - x[:, i]) + c2 * np.random.rand() * (xp[:,ig] - x[:, i])
        x[:, i] = x[:, i] + v[:, i]

    f_plot[g] = np.min(fitness)

#CFPSO
"""
w = 0.6
c1 = 2.1
c2 = 2.05

phi = c1 + c2
k = 2/np.abs(2-phi-np.sqrt(phi**2-4*phi))

for g in range(G):
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plot_contour(f, x, xl, xu)

    for i in range(N):
        fx = f(x[0, i], x[1, i])

        if fx < fitness[i]:
            xp[:, i] = x[:, i]
            fitness[i] = fx

        ig = np.argmin(fitness)

    for i in range(N):
        v[:, i] = K * v[:, i] + c1 * np.random.rand() * (xp[:, i] - x[:, i]) + \
            c2 * np.random.rand() * (xp[:, ig] - x[:, i])
        x[:, i] = x[:, i] + v[:, i]

    f_plot[g] = np.min(fitness)
"""

#WAPSO
"""
for g in range(G):
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plot_contour(f, x, xl, xu)

    for i in range(N):
        fx = f(x[0, i], x[1, i])

        if fx < fitness[i]:
            xp[:, i] = x[:, i]
            fitness[i] = fx

    ig = np.argmin(fitness)
    w = w_max - (g / G) * (w_max - w_min)


    for i in range(N):
        v[:, i] = w * v[:, i] + c1 * np.random.rand() * (xp[:, i] - x[:, i]) + c2 * np.random.rand() * (xp[:, ig] - x[:, i])
        x[:, i] = x[:, i] + v[:, i]

    f_plot[g] = np.min(fitness)
"""

print("Mínimo global en x=", xp[0, ig], " y=",xp[1, ig], " f(x,y)=", f(xp[0, ig], xp[1, ig]))
plot_surf(f, x, xl, xu, ig)
plt.plot(range(G), f_plot)
