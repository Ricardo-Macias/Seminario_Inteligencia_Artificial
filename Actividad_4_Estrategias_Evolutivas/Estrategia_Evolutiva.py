import matplotlib.pyplot as plt
import numpy as np
from plot_Contour import plot_contour
from plot_surf import plot_suft


def Recombination(x1, x2):
    n = np.size(x1)
    y = np.zeros(n)
    Tipo_Recomb = 0  # sexual discreta -> 1, sexual intermedia -> 0

    for d in range(n):
        if Tipo_Recomb == 1:
            if np.random.randint(0, 2):
                y[d] = x1[d]
            else:
                y[d] = x2[d]
        else:
            y[d] = 0.5 * (x1[d] + x2[d])
    return y


f = lambda x,y : x * np.exp(-x**2 - y**2)

#f = lambda x, y: (x-2)**2 + (y-2)**2
xl = np.array([-4, -4])
xu = np.array([4, 4])

G = 50
D = 2
mu = 30
l = 60

x = np.zeros((D, mu+l))
sigma = np.zeros((D, mu+l))
fitness = np.zeros(mu+l)

p_plot = np.zeros(G)

for i in range(mu):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    sigma[:, i] = 0.5 * np.random.rand(D)
    fitness[i] = f(x[0, i], x[1, i])

for g in range(G):

    for i in range(l):

        r1 = np.random.randint(mu)
        r2 = r1

        while r2 == r1:
            r2 = np.random.randint(mu)

        x[:, mu+i] = Recombination(x[:, r1], x[:, r2])
        sigma[:, mu+i] = Recombination(sigma[:, r1], sigma[:, r2])

        r = np.random.normal(0, sigma[:, mu+i], D)
        x[:, mu+i] = x[:, mu+i] + r
        fitness[mu+i] = f(x[0, mu+i], x[1, mu+i])

    Idx = np.argsort(fitness)
    x = x[:, Idx]
    sigma = sigma[:, Idx]
    fitness = fitness[Idx]

    p_plot[g] = np.min(fitness)

Idx_b = np.argmin(fitness)
print("Mínimo en x=", x[0, Idx_b], " y=", x[1, Idx_b],
      " f(x,y)=", f(x[0, Idx_b], x[1, Idx_b]))
plot_suft(f, x, xl, xu)
plot_contour(f, x[:, 0:mu], xl, xu)

plt.plot(p_plot)
plt.show()
