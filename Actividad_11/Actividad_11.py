import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")
import matplotlib.pyplot as plt
import numpy as np
from Actividad_8_Funciones_de_penalizacion import Plot_Surf
from Actividad_8_Funciones_de_penalizacion import Plot_Contour
import math

Griewank = lambda x,y: ((x**2/4000)+(y**2/4000)) - (np.cos(x) * np.cos(y/np.sqrt(2))) + 1
Rastrigin = lambda x,y:10*2 + x**2 + y**2 - 10 * np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
Sphere = lambda x,y: x**2 + y**2


xl = np.array([-5, -5])
xu = np.array([5, 5])

D = 2
G = 150
N = 50

beta0 = 0.5  # 0.1, 0.6
gamma = 0.6  # 0.1, 0.8

alpha = 2.5  # 0.5, 1.5
delta = 0.95  # 0.01, 0.95

x = np.zeros((D, N))
I = np.zeros(N)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    I[i] = Griewank(x[0, i], x[1, i])

for g in range(G):

    for i in range(N):
        for j in range(N):
            if I[j] < I[i]:
                rij = np.linalg.norm(x[:, i] - x[:, j])
                r = np.random.rand(D)

                x[:, i] = x[:, i] + beta0 * \
                    np.exp(-gamma * (rij ** 2)) * \
                    (x[:, j] - x[:, i]) + alpha * (r - 0.5)

                I[i] = Griewank(x[0, i], x[1, i])

    alpha = delta * alpha

igb = np.argmin(I)

print("Mínimo global en x=", x[0, igb], " y=",
      x[1, igb], " f(x,y)=", Griewank(x[0, igb], x[1, igb]))
Plot_Contour.plot_contour(Griewank, x, xl, xu)
Plot_Surf.plot_surf(Griewank, x, xl, xu, igb)
