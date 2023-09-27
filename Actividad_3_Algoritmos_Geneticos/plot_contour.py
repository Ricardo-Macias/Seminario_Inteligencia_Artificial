import matplotlib.pyplot as plt
import numpy as np


def plot_contour(f, x, xl, xu):
    plt.clf()
    X = np.arange(xl[0], xu[0], 0.25)
    Y = np.arange(xl[1], xu[1], 0.25)
    X, Y = np.meshgrid(X, Y)

    # plt.figure()
    plt.contourf(X, Y, f(X, Y))
    # plt.contour(X, Y, Z(X,Y))
    # plt.colorbar()

    plt.title("Contour")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(x[0], x[1], marker="o",
                facecolors='none', edgecolors='r', s=120)

    plt.show()
    plt.pause(.05)
