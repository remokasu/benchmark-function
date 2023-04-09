import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_function(xs: np.ndarray) -> np.float64:
    y = np.sum(xs**2)
    return y

def plot_sphere_function(lower: float, upper: float, num_points: int):
    x0 = np.linspace(lower, upper, num_points)
    x1 = np.linspace(lower, upper, num_points)
    y = np.array([sphere_function(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1]).reshape(num_points, num_points)

    X0, X1 = np.meshgrid(x0, x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(X0, X1, y, color='blue')

    plt.show()

if __name__ == "__main__":
    plot_sphere_function(-2, 2, 32)
