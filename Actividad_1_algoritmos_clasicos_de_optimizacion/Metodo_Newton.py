import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from IPython import display
from time import sleep


def plot_2d(f,x,xl,xu):
    plt.clf()
    X = np.arange(xl, xu, 0.25)
    Y = f(X)

    plt.plot(X,Y)
    plt.plot(x, f(x), marker="o", c="r", markersize=10)

    plt.title("Metodo de Newton")
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.show(block=False)
    plt.pause(.5)


f = lambda x: (20-2*x)*(20-2*x)*x
fp = lambda x: 12*x**2 - 160*x + 400 
fpp = lambda x: 24*x - 160

xl = 0
xu = 10
xi = 8

for i in range(50):
    xi = xi - (fp(xi) / fpp(xi))
    display.display(plt.gcf())
    display.clear_output(wait=True)

    plot_2d(f, xi, xl, xu)

    sleep(0.5)

if fpp(xi) > 0:
    print("Minimo en x= ", xi, " f(x)= ", f(xi))
else:
    print("Maximo en x= ",xi, " f(x)= ",f(xi))


