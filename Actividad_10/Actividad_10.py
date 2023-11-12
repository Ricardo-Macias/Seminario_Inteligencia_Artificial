import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")
import matplotlib.pyplot as plt
import numpy as np
from Actividad_8_Funciones_de_penalizacion import Plot_Contour
from Actividad_8_Funciones_de_penalizacion import Plot_Surf
import math


def f(x, y): return (x-2)**2 + (y-2)**2
# f = lambda x, y: 10*2 + x**2 + y**2 - 10*np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)


xl = np.array([-5, -5])
xu = np.array([5, 5])

D = 2
N = 30
G = 50

p = 0.8
l = 1.5
sigma2 = (((math.gamma(1+l))/(l*math.gamma((1+l)/2))) *
          ((np.sin((np.pi*l)/2))/(2 ** ((l-1)/2)))) ** (1/l)

x = np.zeros((D, N))
fitness = np.zeros(N)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = f(x[0, i], x[1, i])

for t in range(G):

    igb = np.argmin(fitness)

    for i in range(N):
        if np.random.rand() < p:
            u = np.random.normal(0, sigma2, D)
            v = np.random.normal(0, 1, D)
            L = u / (np.abs(v) ** (1 / l))

            y = x[:, i] + L * (x[:, igb] - x[:, i])
        else:
            j = i
            while j == i:
                j = np.random.randint(N)

            k = j
            while k == j and k != i:
                k = np.random.randint(N)

            y = x[:, i] + np.random.rand() * (x[:, j] - x[:, k])

        fy = f(y[0], y[1])

        if fy < fitness[i]:
            x[:, i] = y
            fitness[i] = fy


igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",
      x[1, igb], " f(x,y)=", f(x[0, igb], x[1, igb]))
Plot_Contour.plot_contour(f, x, xl, xu)
Plot_Surf.plot_surf(f, x, xl, xu, igb)

