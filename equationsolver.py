import numpy as np
from equation import *
from newton import Newton
from parametrized import Parametrized
from typing import Tuple


class Equation:
    def __init__(self, sections: Sections,
                 parameters: Parameters,
                 boundary: BoundaryConditions):
        self.sections = sections
        self.parameters = parameters
        self.boundary = boundary

    def solve(self,
              steps: Tuple[float, float],
              method: SolutionMethods,
              epsilon: float = 1e-10):
        h, tau = steps
        a, b = self.sections.x
        t0, T = self.sections.t
        Nt = int((T - t0) / tau) + 1
        solution = self.__create_boundary(steps)
        for layer in range(Nt - 1):
            start_vector = solution[layer]
            if method == SolutionMethods.NEWTON:
                x = Newton(self.sections, self.parameters, self.boundary, steps, epsilon) \
                    .solution(layer, start_vector)
            elif method == SolutionMethods.PARAMETRIZED:
                x = Parametrized(self.sections, self.parameters, self.boundary, steps) \
                    .solution(layer, start_vector)
            else:
                raise Exception('Unknown solution method')
            solution[layer + 1] = x
        return solution

    def __create_boundary(self, steps: Tuple[float, float]):
        a, b = self.sections.x
        t0, T = self.sections.t
        h, tau = steps
        nx = int((b - a) / h) + 1
        nt = int((T - t0) / tau) + 1
        matrix = np.zeros(shape=(nt, nx))
        matrix[0] = self.boundary.initial(np.linspace(a, b, nx))
        return matrix
