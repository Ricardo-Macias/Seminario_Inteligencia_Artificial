import matplotlib.pyplot as plt
import numpy as np
from Actividad_1_algoritmos_clasicos_de_optimizacion.Metodo_Gradiente_Descendiente import plot_surf
from Actividad_1_algoritmos_clasicos_de_optimizacion.Metodo_Gradiente_Descendiente import plot_contour
from IPython import display

f = lambda x, y: (x-2)**2 + (y-2)**2 #Cambiar funcion

xl = np.array([-5,-5])
xu = np.array([5,5])

G = 500
f_plot = np.zeros(G)
mu, sigma = 0, 0.2

x = np.array([4, 4])

for i in range(G):
    y = xl + (xu-xl) * np.random.random(2)

    display.display(plt.gcf())
    display.clear_output(wait=True)
    plot_contour(f, x, y, xl, xu)

    if f(y[0], y[1]) < f(x[0], x[1]):
        x = y

    f_plot[i] = f(x[0], x[1])

plot_surf(f, x, xl, xu)
print("Mínimo global en x=", x[0], " y=", x[1], " f(x,y)=", f(x[0], x[1]))

plt.plot(range(G), f_plot)
plt.title("Convergencia")
