import matplotlib.pyplot as plt
import numpy as np
from Plot_Contour import *
from Plot_Surf import *
from IPython import display

#FUNCIONES OBJETIVO
Griewank = lambda x,y: ((x**2/4000)+(y**2/4000))-(np.cos(x) * np.cos(y/np.sqrt(2))) + 1
Rastrigin = lambda x,y: 10*2 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
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
fitness = np.zeros(N)

f_plot = np.zeros(G)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    xp[:,i] = x[:,i]
    v[:,i] = 0.5 * np.random.randn(D)
    # --------------> Funcion Objetivo
    fitness[i] = Sphere(x[0, i], x[1, i])

#-----------------> PSO <--------------
def PSO():
    for g in range(G):

        for i in range(N):
            fx = Sphere(x[0, i], x[1, i])  # --------------> Funcion Objetivo

            if fx < fitness[i]:
                xp[:,i] = x[:, i]
                fitness[i] = fx
            
            ig = np.argmin(fitness)
        
        for i in range(N):
            v[:,i] = w * v[:, i] + c1 * np.random.rand() * (xp[:,i] - x[:, i]) + c2 * np.random.rand() * (xp[:,ig] - x[:, i])
            x[:, i] = x[:, i] + v[:, i]

        f_plot[g] = np.min(fitness)
    return ig

# --------------------> CFPSO <-------------------

def CFPSO():
    w = 0.6
    c1 = 2.1
    c2 = 2.05

    phi = c1 + c2
    k = 2/np.abs(2-phi-np.sqrt(phi**2-4*phi))

    for g in range(G):

        for i in range(N):
            fx = Sphere(x[0, i],x[1,i]) #--------------> Funcion Objetivo

            if fx < fitness[i]:
                xp[:, i] = x[:, i]
                fitness[i] = fx

            ig = np.argmin(fitness)

        for i in range(N):
            v[:, i] = k * v[:, i] + c1 * np.random.rand() * (xp[:, i] - x[:, i]) + \
                c2 * np.random.rand() * (xp[:, ig] - x[:, i])
            x[:, i] = x[:, i] + v[:, i]

        f_plot[g] = np.min(fitness)
    return ig


# -------------------> WAPSO <------------------------

"""
for g in range(G):

    for i in range(N):
        fx = Griewank(x[0, i], x[1, i])  # --------------> Funcion Objetivo

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
ig = CFPSO()
print("Mínimo global en x=", xp[0, ig], " y=", xp[1, ig], " f(x,y)=", Sphere(xp[0, ig], xp[1, ig]))  # --------------> Funcion Objetivo
plot_contour(Sphere, x, xl, xu)  # --------------> Funcion Objetivo
plot_surf(Sphere, x, xl, xu, ig)  # --------------> Funcion Objetivo
plt.plot(range(G), f_plot)
plt.title("Convergencia")
plt.draw()
plt.show()
