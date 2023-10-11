

import matplotlib.pyplot as plt
import numpy as np
#from Plot_Surf import *
#from Plot_Contour import *
from IPython import display

#----------------------------------------------------
#----------> FUNCION OBJETIVO <----------------------
#----------------------------------------------------


# ----------------------------------------------------
# ---------------> PARAMETROS <-----------------------
# ----------------------------------------------------

xl = np.array([-5,-5])
xu = np.array([5,5])

G = 30
N = 50
D = 2

F = 0.6
CR = 0.9

x = np.zeros((D,N))
fitness = np.zeros(N)

for i in range(N):
    x[:,i] = xl, + (xu - xl) * np.random.rand(D)
    fitness[i] = f(x[0,i],x[1,i]) #Funcion objetivo

fx_plot = np.zeros(G)


# ----------------------------------------------------
# ---------------------> DE <-------------------------
# ----------------------------------------------------
def DE():
    for n in range(G):
        display.display(plt.gcf())
        display.clear_output(wait=True)
        plot_contour(f, x, xl, xu)

        for i in range(N):
            # Mutación
            r1 = i
            while r1 == i:
                r1 = np.random.randint(N)

            r2 = r1
            while r2 == r1 or r2 == i:
                r2 = np.random.randint(N)

            r3 = r2
            while r3 == r2 or r3 == r1 or r3 == i:
                r3 = np.random.randint(N)

            v = x[:, r1] + F * (x[:, r2] - x[:, r3])

            # Recombinación
            u = np.zeros(D)

            for j in range(D):
                r = np.random.rand()

                if r <= CR:
                    u[j] = v[j]
                else:
                    u[j] = x[j, i]

            # Selección
            fitness_u = f(u[0], u[1])

            if fitness_u < fitness[i]:
                x[:, i] = u
                fitness[i] = fitness_u

        fx_plot[n] = np.min(fitness)


# ----------------------------------------------------
# ----------------> DE BEST 1 BIN <-------------------
# ----------------------------------------------------
def best():
    for n in range(G):
        display.display(plt.gcf())
        display.clear_output(wait=True)
        plot_contour(f, x, xl, xu)

        for i in range(N):
            # Mutación
            r1 = i
            while r1 == i:
                r1 = np.random.randint(N)

            r2 = r1
            while r2 == r1 or r2 == i:
                r2 = np.random.randint(N)

            best = np.argmin(fitness)

            v = x[:, best] + F * (x[:, r1] - x[:, r2])

            # Recombinación
            u = np.zeros(D)
            k = np.random.randint(D)

            for j in range(D):
                r = np.random.rand()

                if r <= CR or j == k:
                    u[j] = v[j]
                else:
                    u[j] = x[j, i]

            # Selección
            fitness_u = f(u[0], u[1])

            if fitness_u < fitness[i]:
                x[:, i] = u
                fitness[i] = fitness_u

        fx_plot[n] = np.min(fitness)

# ----------------------------------------------------
# -----------> DE CURRENT TO RAND 1 EXP <-------------
# ----------------------------------------------------
def current_to_rand():
    for n in range(G):
        display.display(plt.gcf())
        display.clear_output(wait=True)
        plot_contour(f, x, xl, xu)

        for i in range(N):
            # Mutación
            r1 = i
            while r1 == i:
                r1 = np.random.randint(N)

            r2 = r1
            while r2 == r1 or r2 == i:
                r2 = np.random.randint(N)

            r3 = r2
            while r3 == r2 or r3 == r1 or r3 == i:
                r3 = np.random.randint(N)

            v = x[:, i] + F * (x[:, r1] - x[:, i]) + F * (x[:, r2] - x[:, r3])

            # Recombinación
            u = x[:, i].copy()  # vector de prueba
            j = np.random.randint(D)
            L = 1

            u[j] = v[j]
            while np.random.rand() <= CR and L < D:
                u[j] = v[j]
                j = np.mod(j, D)
                L = L + 1

            # Selección
            fitness_u = f(u[0], u[1])

            if fitness_u < fitness[i]:
                x[:, i] = u
                fitness[i] = fitness_u

        fx_plot[n] = np.min(fitness)

igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",
      x[1, igb], " f(x,y)=", f(x[0, igb], x[1, igb]))
plot_surf(f, x, xl, xu, igb)
plt.plot(fx_plot)
