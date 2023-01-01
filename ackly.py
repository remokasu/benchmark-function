import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# name: Ackley function
# search area: −32.768 <= xi <= 32.768
# best solution: f(0, 0) = 0


def func(xs: np.ndarray) -> np.float64:
    n = len(xs)
    y = 20 - 20*np.exp(-0.2*np.sqrt(1/n*np.sum(xs**2))) + np.e - np.exp(1/n*np.sum(np.cos(2*np.pi*xs)))
    return y


if __name__ == "__main__":
    x0 = np.linspace(-32.768, 32.768, 32)
    x1 = np.linspace(-32.768, 32.768, 32)
    y = np.array([func(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1])
    y = y.reshape(32, 32)

    X0, X1 = np.meshgrid(x0, x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X0, X1, y, color='blue')
