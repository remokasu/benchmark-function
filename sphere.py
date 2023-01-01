import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# name: sphere function
# search area: −inf <= xi <= inf
# best solution: f(0, 0) = 0


def func(xs: np.ndarray) -> np.float64:
    y = np.sum(xs**2)
    return y


if __name__ == "__main__":
    lower = -2
    upper = 2
    x0 = np.linspace(lower, upper, 32)
    x1 = np.linspace(lower, upper, 32)
    y = np.array([func(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1])
    y = y.reshape(32, 32)

    X0, X1 = np.meshgrid(x0, x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X0, X1, y, color='blue')
