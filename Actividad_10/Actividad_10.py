import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")
import matplotlib.pyplot as plt
import numpy as np
from Actividad_8_Funciones_de_penalizacion import Plot_Contour
from Actividad_8_Funciones_de_penalizacion import Plot_Surf
import math


Griewank = lambda x,y: ((x**2/4000)+(y**2/4000)) - (np.cos(x) * np.cos(y/np.sqrt(2))) + 1
Rastrigin = lambda x,y:10*2 + x**2 + y**2 - 10 * np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
Sphere = lambda x,y: x**2 + y**2


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
f_plot = np.zeros(G)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = Griewank(x[0, i], x[1, i])

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

        fy = Griewank(y[0], y[1])

        if fy < fitness[i]:
            x[:, i] = y
            fitness[i] = fy
    
    f_plot[t] = np.min(fitness)


igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",
      x[1, igb], " f(x,y)=", Griewank(x[0, igb], x[1, igb]))
Plot_Contour.plot_contour(Griewank, x, xl, xu)
Plot_Surf.plot_surf(Griewank, x, xl, xu, igb)
plt.plot(range(G), f_plot)
plt.title("Convergencia")
plt.draw()
plt.show()

