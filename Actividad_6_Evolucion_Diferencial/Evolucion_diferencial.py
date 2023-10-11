import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")

import matplotlib.pyplot as plt
import numpy as np
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Contour
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Surf
from IPython import display

#----------------------------------------------------
#----------> FUNCION OBJETIVO <----------------------
#----------------------------------------------------

Griewank = lambda x,y: ((x**2/4000)+(y**2/4000))-(np.cos(x)* np.cos(y/np.sqrt(2))) + 1
Rastrigin = lambda x,y: 10*2 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
Sphere = lambda x,y: x**2 + y**2

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
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = Sphere(x[0, i], x[1, i])  # Funcion objetivo

fx_plot = np.zeros(G)


# ----------------------------------------------------
# ---------------------> DE <-------------------------
# ----------------------------------------------------
def DE():
    for n in range(G):

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
            fitness_u = Sphere(u[0], u[1])

            if fitness_u < fitness[i]:
                x[:, i] = u
                fitness[i] = fitness_u

        fx_plot[n] = np.min(fitness)


# ----------------------------------------------------
# ----------------> DE rand 2 exp <-------------------
# ----------------------------------------------------
def rand():
    for n in range(G):

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

            r4 = r3
            while r4 == r3 or r4 == r2 or r4 == r1 or r4 == i:
                r4 = np.random.randint(N)

            r5 = r4
            while r5 == r4 or r5== r3 or r5 == r2 or r5 == r1 or r5 == i:
                r5 = np.random.randint(N)

            v = x[:, r1] + F * (x[:, r2] - x[:, r3]) + F * (x[:,r4] - x[:,r5])

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
            fitness_u = Sphere(u[0], u[1])

            if fitness_u < fitness[i]:
                x[:, i] = u
                fitness[i] = fitness_u

        fx_plot[n] = np.min(fitness)

# ----------------------------------------------------
# -----------> DE CURRENT TO best 2 exp <-------------
# ----------------------------------------------------
def current_to_best():
    for n in range(G):

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
            
            r4 = r3
            while r4 == r3 or r4 == r2 or r4 == r1 or r4 == i:
                r3 = np.random.randint(N)

            best = np.argmin(fitness)

            v = x[:, i] + F * ( x[:,best] - x[:,i] + F *(x[:, r1] - x[:, r2]) + F * (x[:, r3] - x[:, r4]))

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
            fitness_u = Sphere(u[0], u[1])

            if fitness_u < fitness[i]:
                x[:, i] = u
                fitness[i] = fitness_u

        fx_plot[n] = np.min(fitness)

rand()
igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",
      x[1, igb], " f(x,y)=", Sphere(x[0, igb], x[1, igb]))
Plot_Contour.plot_contour(Sphere, x, xl, xu)
Plot_Surf.plot_surf(Sphere, x, xl, xu, igb)
plt.plot(fx_plot)
plt.title("Convergencia")
plt.draw()
plt.show()
