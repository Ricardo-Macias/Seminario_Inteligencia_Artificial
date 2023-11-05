import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")
import matplotlib.pyplot as plt
import numpy as np
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Contour
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Surf


Griewank = lambda x,y: ((x**2/4000)+(y**2/4000)) - (np.cos(x) * np.cos(y/np.sqrt(2))) + 1
Rastrigin = lambda x,y:10*2 + x**2 + y**2 - 10 * np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
Sphere = lambda x,y: x**2 + y**2
McCormick = lambda x,y: np.sin(x+y) + (x-y) ** 2 - 1.5*x + 2.5*y + 1

xl = np.array([-5, -5])
xu = np.array([5, 5])

D = 2
N = 30
G = 100

x = np.zeros((D, N))
fitness = np.zeros(N)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = Griewank(x[0, i], x[1, i])

for g in range(G):

    for i in range(N):
        # Teacher phase
        t = np.argmin(fitness)
        Tf = np.random.randint(2)
        c = np.zeros(D)

        for j in range(D):
            x_mean = np.mean(x[j, :])
            r = np.random.rand()

            c[j] = x[j, i] + r * (x[j, t] - Tf * x_mean)

        fc = Griewank(c[0], c[1])

        if fc < fitness[i]:
            x[:, i] = c
            fitness[i] = fc

        # Learner phase
        k = i
        while k == i:
            k = np.random.randint(N)

        c = np.zeros(D)

        if fitness[i] < fitness[k]:
            for j in range(D):
                r = np.random.rand()
                c[j] = x[j, i] + r * (x[j, i] - x[j, k])
        else:
            for j in range(D):
                r = np.random.rand()
                c[j] = x[j, i] + r * (x[j, k] - x[j, i])

        fc = f(c[0], c[1])

        if fc < fitness[i]:
            x[:, i] = c
            fitness[i] = fc

igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",x[1, igb], " f(x,y)=", Griewank(x[0, igb], x[1, igb]))
Plot_Surf.plot_surf(Griewank, x, xl, xu, igb)
Plot_Contour.plot_contour(Griewank, x, xl, xu)
