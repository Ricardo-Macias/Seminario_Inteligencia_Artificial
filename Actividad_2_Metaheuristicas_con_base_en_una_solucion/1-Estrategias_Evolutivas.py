import matplotlib.pyplot as plt
import numpy as np
from plot_contour import *
from plot_surf import *
from IPython import display


#f = lambda x, y : x * np.exp(-x**2-y**2) #Primera funcion
f = lambda x, y: x**2 + y**2 #Segunda funcion


xl = np.array([-5, -5])
xu = np.array([5, 5])

G = 500
f_plot = np.zeros(G)
mu, sigma = 0, 0.2

x = xl + (xu-xl) * np.random.random(2)
f_plot = np.zeros(G)

for i in range(G):
    r = np.random.normal(mu, sigma, 2)
    y = x + r

    if f(y[0], y[1]) < f(x[0], x[1]):
        x = y

    # display.display(plt.gcf())
    # display.clear_output(wait=True)
    # plot_contour(f, x, y, xl, xu)
    f_plot[i] = f(x[0], x[1])

print("Mínimo global en x=", x[0], " y=", x[1], " f(x,y)=", f(x[0], x[1]))
plot_contour(f, x, y, xl, xu)
plot_surf(f, x, xl, xu)

plt.plot(range(G), f_plot)
plt.title("Convergencia")
plt.draw()
plt.show()
