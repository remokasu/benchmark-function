import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from benchmark import BF

BF.FUNCTION_RANGES.keys()
function_names = BF.FUNCTION_RANGES.keys()

fig, axs = plt.subplots(10, 4, figsize=(15, 20), subplot_kw={'projection': '3d'})
fig.tight_layout()

for idx, func_name in enumerate(function_names):
    row, col = divmod(idx, 4)
    bf = BF(func_name)

    lower = bf.FUNCTION_RANGES[func_name]["lower"]
    upper = bf.FUNCTION_RANGES[func_name]["upper"]

    x0 = np.linspace(lower[0], upper[0], 32)
    x1 = np.linspace(lower[1], upper[1], 32)
    y = np.array([bf.evaluate(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1]).reshape(32, 32)

    X0, X1 = np.meshgrid(x0, x1)

    axs[row, col].plot_surface(X0, X1, y, cmap='viridis', linewidth=0, antialiased=False, alpha=0.8)
    axs[row, col].set_title(func_name)
    axs[row, col].axis('off')

for i in range(len(function_names), 28):
    row, col = divmod(i, 4)
    fig.delaxes(axs[row, col])

plt.savefig("benchmark_functions_overview.png", dpi=100)
