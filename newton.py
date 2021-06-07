import numpy as np
from scipy.linalg import solve_banded
from typing import List, Tuple
from equation import *

class Newton:
    def __init__(self, sections: Sections,
                 parameters: Parameters,
                 boundary: BoundaryConditions,
                 steps: Tuple[float, float],
                 epsilon: float = 1e-10):
        self.sections = sections
        self.parameters = parameters
        self.boundary = boundary
        self.steps = steps
        self.epsilon = epsilon

    def solution(self,
                 layer: int,
                 start_vector: List[float]):
        x0 = start_vector
        function = self.__function(layer, x0, x0)
        jacobian = self.__jacobian(x0)
        delta = solve_banded((1, 1), jacobian, -function,
                             overwrite_ab=True, overwrite_b=True)
        x1 = x0 + delta
        while np.max(np.abs(x1 - x0)) >= self.epsilon:
            x0 = x1
            function = self.__function(layer, x0, start_vector)
            jacobian = self.__jacobian(x0)
            delta = solve_banded((1, 1), jacobian, -function,
                                 overwrite_ab=True, overwrite_b=True)
            x1 = x0 + delta
        return x1

    def __function(self,
                   layer: int,
                   x: List[float],
                   previous_x: List[float]):
        a, b = self.sections.x
        t0, T = self.sections.t
        g0, g1 = self.boundary.left, self.boundary.right
        k, cp, sigma = self.parameters.k, self.parameters.cp, \
                       self.parameters.sigma
        h, tau = self.steps
        nx = int((b - a) / h) + 1
        nt = int((T - t0) / tau) + 1
        x_nodes = np.linspace(a, b, nx)
        t_nodes = np.linspace(t0, T, nt)
        assert x_nodes[-1] == b and t_nodes[-1] == T, f'Invalid spliting'
        t_next = t_nodes[layer + 1]
        gamma = self.parameters.k * tau / np.square(h)
        fi = lambda i: (self.parameters.cp * previous_x[i] +
                        tau * self.parameters.function(x_nodes[i], t_next))

        first = [-k / h * (x[1] - x[0]) + sigma * x[0] * np.fabs(np.power(x[0], 3)) -
                 g0(t_next) - h / 2 * (self.parameters.function(a, t_next) -
                                       cp / tau * (x[0] - previous_x[0]))]

        mid = [gamma * x[i + 1] - (cp + 2 * gamma) * x[i] + gamma * x[i - 1] + fi(i)
               for i in range(1, nx - 1)]

        last = [k / h * (x[-1] - x[-2]) + sigma * x[-1] * np.fabs(np.power(x[-1], 3)) -
                g1(t_next) - h / 2 * (self.parameters.function(b, t_next) -
                                      cp / tau * (x[-1] - previous_x[-1]))]

        return np.concatenate((first, mid, last))

    def __jacobian(self, x: List[float]):
        a, b = self.sections.x
        k, cp, sigma = self.parameters.k, self.parameters.cp, \
                       self.parameters.sigma
        h, tau = self.steps
        nx = int((b - a) / h) + 1
        gamma = k * tau / np.square(h)

        first = [0, -k / h] + [gamma] * (nx - 2)
        mid = [k / h + 4 * sigma * np.fabs(np.power(x[0], 3)) + h * cp / (2 * tau)] + \
              ([-(cp + 2 * gamma)] * (nx - 2)) + \
              [k / h + 4 * sigma * np.fabs(np.power(x[-1], 3)) + h * cp / (2 * tau)]
        last = [gamma] * (nx - 2) + [-k / h, 0]

        return np.row_stack((first, mid, last))
