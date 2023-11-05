import sys
sys.path.append(
    "D:\\Archivos\\Practicas\\7_Semestre\\Seminario_Inteligencia_Artificial")
import matplotlib.pyplot as plt
import numpy as np
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Contour
from Actividad_5_Optimizacion_por_enjambre_de_particulas import Plot_Surf

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
    return z

Griewank = lambda x,y: ((x**2/4000)+(y**2/4000)) - (np.cos(x) * np.cos(y/np.sqrt(2))) + 1
penalty_Griewank = lambda x,xl,xu: Griewank(x[0],x[1]) + 1000 * penalty(x,xl,xu)

Rastrigin = lambda x,y:10*2 + x**2 + y**2 - 10 * np.cos(2*np.pi*x) - 10*np.cos(2*np.pi*y)
penalty_Rastrigin = lambda x,xl,xu:  Rastrigin(x[0],x[1]) + 1000 * penalty(x,xl,xu)

Sphere = lambda x,y: x**2 + y**2
penalty_Sphere = lambda x,xl,xu: Sphere(x[0],x[1]) + 1000 * penalty(x,xl,xu)

McCormick = lambda x,y: np.sin(x+y) + (x-y) ** 2 - 1.5*x + 2.5*y + 1
penalty_McCormick = lambda x,xl,xu: McCormick(x[0],x[1]) + 1000 * penalty(x,xl,xu)

xl = np.array([-5, -5])
xu = np.array([5, 5])

D = 2
N = 30
G = 100

x = np.zeros((D, N))
fitness = np.zeros(N)
f_plot = np.zeros(G)

for i in range(N):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    fitness[i] = penalty_Griewank(x[:, i], xl,xu)

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

        fc = penalty_Griewank(c,xl,xu)

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

        fc = penalty_Griewank(c,xl,xu)

        if fc < fitness[i]:
            x[:, i] = c
            fitness[i] = fc
    
    f_plot[g] = np.min(fitness)

igb = np.argmin(fitness)

print("Mínimo global en x=", x[0, igb], " y=",x[1, igb], " f(x,y)=", penalty_Griewank(x[:, igb],xl,xu))
Plot_Contour.plot_contour(penalty_Griewank, x, xl, xu)
Plot_Surf.plot_surf(penalty_Griewank, x, xl, xu, igb)
plt.plot(range(G), f_plot)
plt.title("Convergencia")
plt.draw()
plt.show()
