import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
from equationsolver import Equation

class Plot:
    def __init__(self, equation: Equation):
        self.equation = equation

    def draw(self,
             steps: Tuple[float, float],
             title: str,
             kind: str = 'plot',
             values: List[List[float]] = None,
             real_solution: Callable[[float, float], float] = None,
             save_to: str = None):
        a, b = self.equation.sections.x
        t, T = self.equation.sections.t
        h, tau = steps
        nx = int((b - a) / h) + 1
        nt = int((T - t) / tau) + 1
        figure = plt.figure(figsize=(6, 6))
        axes = figure.add_subplot(1, 1, 1, projection='3d')
        axes.set_xlabel('x')
        axes.set_ylabel('t')

        X = np.linspace(a, b, nx)
        T = np.linspace(t, T, nt)
        X, T = np.meshgrid(X, T)

        if kind == 'error' and real_solution is not None and values is not None:
            real = real_solution(X, T)
            error = np.abs(values - real)
            values = error
        elif real_solution is not None and values is None:
            assert values is None, 'Redefinition of "U"'
            values = real_solution(X, T)
            if isinstance(values, float): # на случай, если решение константа
                values = np.ones(shape=X.shape) * values
        surface = axes.plot_surface(X, T, values, cmap='inferno')
        plt.title(title)
        if save_to is not None:
            plt.savefig(save_to)
        # plt.show()
