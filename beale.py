import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# name: Beale function
# search area: -4.5 <= xi <= 4.5
# best solution: f(3, 0.5) = 0


def func(xs: np.ndarray) -> np.float64:
    x1 = xs[0]
    x2 = xs[1]
    y = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

    return y


if __name__ == "__main__":
    lower = -4.5
    upper = 4.5
    x0 = np.linspace(lower, upper, 32)
    x1 = np.linspace(lower, upper, 32)
    y = np.array([func(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1])
    y = y.reshape(32, 32)

    X0, X1 = np.meshgrid(x0, x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X0, X1, y, color='blue')
