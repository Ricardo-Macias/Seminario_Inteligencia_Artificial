import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from IPython import display
from time import sleep

def plot_contour(f,x,xl,xu):
    plt.clf()
    X = np.arange(xl[0], xu[0], 0.25)
    Y = np.arange(xl[1], xu[1], 0.25)
    X, Y = np.meshgrid(X,Y)

    #plt.figure()
    plt.contourf(X,Y, f(X,Y))
    #plt.countour(X, Y, Z(X,Y))
    #plt.colorbar()

    plt.title("Contour")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x[0], x[1], marker='o', c='r', markersize=10, label='Óptimo')
    plt.legend()

    plt.show(block=False)
    plt.pause(.05)


def plot_surf(f, x, xl, xu):
    plt.clf()
    X = np.arange(xl[0], xu[0], 0.25)
    Y = np.arange(xl[1], xu[1], 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    # SURFACE
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    ax.set_title('Surf')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter(x[0], x[1], f(x[0], x[1]), c='r', label='Óptimo', s=120)
    ax.legend()

    plt.show()


def F(x, y):
    return (x-2)**2 + (y-2)**2


def G(x, y):
    return np.array([2*x-4,2*y-4])


xl = [0, 0]
xu = [3, 3]
xi = [0, 1]

h = .1
for i in range(100):
    # Las funciones de display nos ayudan a no estar generando
    # múltiples imágenes.
    display.display(plt.gcf())
    display.clear_output(wait=True)

    plot_contour(F, xi, xl, xu)
    xi = xi - h * G(xi[0], xi[1])
    # sleep(0.05)

plot_surf(F, xi, xl, xu)
print("Mínimo global en:", xi)


