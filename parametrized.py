import numpy as np
from parametrized_solution import SolutionForTwoNonLinear
from equation import *


class Parametrized:
    def __init__(self, sections: Sections,
                 parameters: Parameters,
                 boundary: BoundaryConditions,
                 steps: Tuple[float, float]):
        self.sections = sections
        self.parameters = parameters
        self.boundary = boundary
        self.steps = steps

    def solution(self,
                 layer: int,
                 start_vector: Tuple[float, float]):
        n = len(start_vector)
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

        beta = cp * h / (2 * tau)
        gamma = k * tau / np.square(h)
        B = [0, k / h] + [gamma] * (n - 2)
        C = [k / h + beta] + [cp + 2 * gamma] * (n - 2) + [k / h + beta]
        A = [gamma] * (n - 2) + [k / h, 0]
        ab = np.row_stack((B, C, A))

        fi = lambda i: (cp * start_vector[i] +
                        tau * self.parameters.function(x_nodes[i], t_next))
        right_side = [fi(i) for i in range(1, n - 1)]
        build_function = lambda y, g, j: \
            lambda x: g(t_next) + h / 2 * self.parameters.function(y, t_next) + \
                        beta * start_vector[j] - sigma * x * np.power(np.fabs(x), 3)
        first_func = build_function(a, g0, 0)
        second_func = build_function(b, g1, -1)
        prime = lambda x: -4 * sigma * np.power(np.fabs(x), 3)
        solution = SolutionForTwoNonLinear(ab, right_side,
                                           first_func, prime,
                                           second_func, prime)
        return solution.value(np.array([start_vector[0], start_vector[-1]]))
