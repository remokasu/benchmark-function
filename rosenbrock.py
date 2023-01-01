import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# name: Rosenbrock function
# search area: -5 <= xi <= 5
# best solution: f(1, 1) = 0


def func(xs: np.ndarray) -> np.float64:
    y = 0.0
    for i in range(len(xs)-1):
        y += (100 * (xs[i+1] - xs[i]**2)**2 + (xs[i]-1)**2)
    return y


if __name__ == "__main__":
    lower = -5
    upper = 5
    x0 = np.linspace(lower, upper, 32)
    x1 = np.linspace(lower, upper, 32)
    y = np.array([func(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1])
    y = y.reshape(32, 32)

    X0, X1 = np.meshgrid(x0, x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X0, X1, y, color='blue')
