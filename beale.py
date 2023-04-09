import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def beale_function(xs: np.ndarray) -> np.float64:
    x1 = xs[0]
    x2 = xs[1]
    y = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
    return y

def plot_beale_function(lower: float, upper: float, num_points: int):
    x0 = np.linspace(lower, upper, num_points)
    x1 = np.linspace(lower, upper, num_points)
    y = np.array([beale_function(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1]).reshape(num_points, num_points)

    X0, X1 = np.meshgrid(x0, x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X0, X1, y, color='blue')

    plt.show()

if __name__ == "__main__":
    plot_beale_function(-4.5, 4.5, 32)
