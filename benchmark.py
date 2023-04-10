import argparse

import matplotlib.pyplot as plt
import numpy as np


class BF:
    FUNCTION_RANGES = {
        "ackley": {"lower": [-32.768, -32.768], "upper": [32.768, 32.768]},
        "beale": {"lower": [-4.5, -4.5], "upper": [4.5, 4.5]},
        "rastrigin": {"lower": [-5.12, -5.12], "upper": [5.12, 5.12]},
        "sphere": {"lower": [-5.12, -5.12], "upper": [5.12, 5.12]},
        "rosenbrock": {"lower": [-5, -5], "upper": [10, 10]},
        "eggholder": {"lower": [-512, -512], "upper": [512, 512]},
        "schwefel": {"lower": [-500, -500], "upper": [500, 500]},
        "goldstein_price": {"lower": [-2, -2], "upper": [2, 2]},
        "levi": {"lower": [-10, -10], "upper": [10, 10]},
        "himmelblau": {"lower": [-5, -5], "upper": [5, 5]},
        "booth": {"lower": [-10, -10], "upper": [10, 10]},
        "three_hump_camel": {"lower": [-5, -5], "upper": [5, 5]},
        "bukin": {"lower": [-15, -3], "upper": [-5, 3]},
        "matyas": {"lower": [-10, -10], "upper": [10, 10]},
        "six_hump_camel": {"lower": [-3, -2], "upper": [3, 2]},
        "styblinski_tang": {"lower": [-5, -5], "upper": [5, 5]},
        "michalewicz": {"lower": [0, 0], "upper": [np.pi, np.pi]},
        "easom": {"lower": [-100, -100], "upper": [100, 100]},
        "cross_in_tray": {"lower": [-10, -10], "upper": [10, 10]},
        "griewank": {"lower": [-600, -600], "upper": [600, 600]},
        "drop_wave": {"lower": [-5.12, -5.12], "upper": [5.12, 5.12]},
        "dixon_price": {"lower": [-10, -10], "upper": [10, 10]},
        "zakharov": {"lower": [-5, -5], "upper": [10, 10]},
        "salomon": {"lower": [-100, -100], "upper": [100, 100]},
        "alpine": {"lower": [0, 0], "upper": [10, 10]},
        "xin_she_yang": {"lower": [-2 * np.pi, -2 * np.pi], "upper": [2 * np.pi, 2 * np.pi]},
        "brown": {"lower": [-1, -1], "upper": [4, 4]},
        # Add more functions and their ranges here
    }

    def __init__(self, func_name: str):
        self.func_name = func_name

    def plot(self, num_points: int = 32) -> None:
        if self.func_name not in self.FUNCTION_RANGES:
            raise ValueError("Invalid function name")

        lower = self.FUNCTION_RANGES[self.func_name]["lower"]
        upper = self.FUNCTION_RANGES[self.func_name]["upper"]

        x0 = np.linspace(lower[0], upper[0], num_points)
        x1 = np.linspace(lower[1], upper[1], num_points)
        y = np.array([self.calc(np.array([_x0, _x1])) for _x0 in x0 for _x1 in x1]).reshape(num_points, num_points)

        X0, X1 = np.meshgrid(x0, x1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X0, X1, y, color='blue')

        plt.show()

    def get_function(self):
        try:
            func = getattr(self, f"{self.func_name}_function")
            return func
        except AttributeError:
            raise ValueError(f"Invalid function name: {self.func_name}")

    def calc(self, xs: np.ndarray) -> np.float64:
        try:
            func = getattr(self, f"{self.func_name}_function")
            return func(xs)
        except AttributeError:
            raise ValueError(f"Invalid function name: {self.func_name}")

    def ackley_function(self, xs: np.ndarray) -> np.float64:
        n = len(xs)
        y = 20 - 20*np.exp(-0.2*np.sqrt(1/n*np.sum(xs**2))) + np.e - np.exp(1/n*np.sum(np.cos(2*np.pi*xs)))
        return y

    def beale_function(self, xs: np.ndarray) -> np.float64:
        x1 = xs[0]
        x2 = xs[1]
        y = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2
        return y

    def rastrigin_function(self, xs: np.ndarray) -> np.float64:
        """
        Rastrigin function.
        Bounds: -5.12 <= x_i <= 5.12 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
        """
        n = len(xs)
        A = 10
        y = A * n + np.sum(xs**2 - A * np.cos(2 * np.pi * xs))
        return y

    def sphere_function(self, xs: np.ndarray) -> np.float64:
        """
        Sphere function.
        Bounds: -5.12 <= x_i <= 5.12 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
        """
        y = np.sum(xs**2)
        return y

    def rosenbrock_function(self, xs: np.ndarray) -> np.float64:
        """
        Rosenbrock function.
        Bounds: -5 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (1, 1, ..., 1)
        """
        y = np.sum(100 * (xs[1:] - xs[:-1]**2)**2 + (1 - xs[:-1])**2)
        return y

    def eggholder_function(self, xs: np.ndarray) -> np.float64:
        """
        Eggholder function.
        Bounds: -512 <= x_i <= 512 (for each i)
        Global minimum: f(x*) ≈ -959.6407 at x* ≈ (512, 404.2319)
        """
        x1, x2 = xs
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        y = term1 + term2
        return y

    def schwefel_function(self, xs: np.ndarray) -> np.float64:
        """
        Schwefel function.
        Bounds: -500 <= x_i <= 500 (for each i)
        Global minimum: f(x*) = 0 at x* = (420.9687, 420.9687, ..., 420.9687)
        """
        n = len(xs)
        y = 418.9829 * n - np.sum(xs * np.sin(np.sqrt(np.abs(xs))))
        return y

    def goldstein_price_function(self, xs: np.ndarray) -> np.float64:
        """
        Goldstein-Price function.
        Bounds: -2 <= x_i <= 2 (for each i)
        Global minimum: f(x*) = 3 at x* = (0, -1)
        """
        x1, x2 = xs
        part1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
        part2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
        y = part1 * part2
        return y

    def levi_function(self, xs: np.ndarray) -> np.float64:
        """
        Lévi function (Lévi N.13).
        Bounds: -10 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (1, 1, ..., 1)
        """
        n = len(xs)
        y = 0
        for i in range(n - 1):
            y += (np.sin(3 * np.pi * xs[i]))**2 + (xs[i] - 1)**2 * (1 + (np.sin(3 * np.pi * xs[i+1]))**2) + (xs[n-1] - 1)**2 * (1 + (np.sin(2 * np.pi * xs[n-1]))**2)
        return y

    def himmelblau_function(self, xs: np.ndarray) -> np.float64:
        """
        Himmelblau function.
        Bounds: -5 <= x_i <= 5 (for each i)
        Global minimum: f(x*) = 0 at x* = (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
        """
        x1, x2 = xs
        y = (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
        return y

    def booth_function(self, xs: np.ndarray) -> np.float64:
        """
        Booth function.
        Bounds: -10 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (1, 3)
        """
        x1, x2 = xs
        y = (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
        return y

    def three_hump_camel_function(self, xs: np.ndarray) -> np.float64:
        """
        Three-Hump Camel function.
        Bounds: -5 <= x_i <= 5 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0)
        """
        x1, x2 = xs
        y = 2 * x1 ** 2 - 1.05 * x1 ** 4 + (x1 ** 6) / 6 + x1 * x2 + x2 ** 2
        return y

    def bukin_function(self, xs: np.ndarray) -> np.float64:
        """
        Bukin function (Bukin N.6).
        Bounds: -15 <= x1 <= -5, -3 <= x2 <= 3
        Global minimum: f(x*) = 0 at x* = (-10, 1)
        """
        x1, x2 = xs
        y = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10)
        return y

    def matyas_function(self, xs: np.ndarray) -> np.float64:
        """
        Matyas function.
        Bounds: -10 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0)
        """
        x1, x2 = xs
        y = 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2
        return y

    def six_hump_camel_function(self, xs: np.ndarray) -> np.float64:
        """
        Six-Hump Camel function.
        Bounds: -3 <= x1 <= 3, -2 <= x2 <= 2
        Global minimum: f(x*) = -1.0316 at x* ≈ (0.0898, -0.7126), (-0.0898, 0.7126)
        """
        x1, x2 = xs
        y = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
        return y

    def styblinski_tang_function(self, xs: np.ndarray) -> np.float64:
        """
        Styblinski-Tang function.
        Bounds: -5 <= x_i <= 5 (for each i)
        Global minimum: f(x*) = -39.16599 * n at x* = (-2.903534, -2.903534, ..., -2.903534)
        """
        return np.sum(xs**4 - 16 * xs**2 + 5 * xs) / 2

    def michalewicz_function(self, xs: np.ndarray) -> np.float64:
        """
        Michalewicz function.
        Bounds: 0 <= x_i <= pi (for each i)
        Global minimum: f(x*) varies with the problem dimension.
        """
        m = 10
        n = len(xs)
        return -np.sum(np.sin(xs) * np.sin((np.arange(1, n + 1) * xs**2) / np.pi)**(2 * m))

    def easom_function(self, xs: np.ndarray) -> np.float64:
        """
        Easom function.
        Bounds: -100 <= x_i <= 100 (for each i)
        Global minimum: f(x*) = -1 at x* = (pi, pi)
        """
        x1, x2 = xs
        return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)

    def cross_in_tray_function(self, xs: np.ndarray) -> np.float64:
        """
        Cross-in-tray function.
        Bounds: -10 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = -2.06261 at x* = (1.34941, 1.34941), (-1.34941, 1.34941), (1.34941, -1.34941), (-1.34941, -1.34941)
        """
        x1, x2 = xs
        return -0.0001 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))) + 1)**0.1

    def griewank_function(self, xs: np.ndarray) -> np.float64:
        """
        Griewank function.
        Bounds: -600 <= x_i <= 600 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
        """
        d = len(xs)
        sum_term = np.sum(xs**2 / 4000)
        prod_term = np.prod(np.cos(xs / np.sqrt(np.arange(1, d + 1))))
        return 1 + sum_term - prod_term

    def drop_wave_function(self, xs: np.ndarray) -> np.float64:
        """
        Drop-wave function.
        Bounds: -5.12 <= x_i <= 5.12 (for each i)
        Global minimum: f(x*) = -1 at x* = (0, 0)
        """
        x1, x2 = xs
        return -(1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (0.5 * (x1**2 + x2**2) + 2)

    def dixon_price_function(self, xs: np.ndarray) -> np.float64:
        """
        Dixon-Price function.
        Bounds: -10 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (2**(-(2**i - 2) / (2**i))) for i = 1, 2, ..., n
        """
        d = len(xs)
        i = np.arange(1, d + 1)
        sum_term = np.sum(i * (2 * xs[1:]**2 - xs[:-1])**2)
        return (xs[0] - 1)**2 + sum_term

    def zakharov_function(self, xs: np.ndarray) -> np.float64:
        """
        Zakharov function.
        Bounds: -5 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
        """
        d = len(xs)
        i = np.arange(1, d + 1)
        sum_term1 = np.sum(xs**2)
        sum_term2 = np.sum(0.5 * i * xs)
        return sum_term1 + sum_term2**2 + sum_term2**4

    def salomon_function(self, xs: np.ndarray) -> np.float64:
        """
        Salomon function.
        Bounds: -100 <= x_i <= 100 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
        """
        r = np.sqrt(np.sum(xs**2))
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r

    def alpine_function(self, xs: np.ndarray) -> np.float64:
        """
        Alpine function.
        Bounds: 0 <= x_i <= 10 (for each i)
        Global minimum: f(x*) = 0 at x* = (0, 0, ..., 0)
        """
        return np.sum(np.abs(xs * np.sin(xs) + 0.1 * xs))

    def xin_she_yang_function(self, xs: np.ndarray) -> np.float64:
        """
        Xin-She Yang function.
        Bounds: -2 * pi <= x_i <= 2 * pi (for each i)
        Global minimum: f(x*) = -1 at x* = (0, 0, ..., 0)
        """
        beta = 15
        return np.sum(np.abs(xs)**beta) - np.prod(np.cos(xs)**2)

    def brown_function(self, xs: np.ndarray) -> np.float64:
        """
        Brown function.
        Bounds: -1 <= x_i <= 4 (for each i)
        Global minimum: f(x*) = 0 at x* = (1, 1, ..., 1)
        """
        return np.sum((xs[:-1]**2)**(xs[1:]**2 + 1) + (xs[1:]**2)**(xs[:-1]**2 + 1))


    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark function calculator and plotter.")
    parser.add_argument("--function", required=True, help="Name of the benchmark function.")
    parser.add_argument("--x0", type=float, help="Value of x0 input.")
    parser.add_argument("--x1", type=float, help="Value of x1 input.")
    parser.add_argument("--num_points", type=int, default=32, help="Number of points for plot.")
    args = parser.parse_args()

    func = BF(args.function)

    if args.x0 is not None and args.x1 is not None:
        print(f"{args.function}({args.x0}, {args.x1}) = {func.calc(np.array([args.x0, args.x1]))}")

    func.plot(args.num_points)


