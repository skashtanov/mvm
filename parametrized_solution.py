import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import solve_banded
from typing import Callable, List


class SolutionForTwoNonLinear:
    def __init__(self,
                 ab: List[float],
                 b: List[float],
                 first_func: Callable[[float], float],
                 first_prime: Callable[[float], float],
                 second_func: Callable[[float], float],
                 second_prime: Callable[[float], float]):
        self.__ab = np.copy(ab)
        self.__b = np.copy(b)
        self.__first_func = first_func
        self.__first_prime = first_prime
        self.__second_func = second_func
        self.__second_prime = second_prime

    def value(self, start_vector=np.zeros(2)):
        v, u, w = self.__solve_subsystems()
        B, C, A = self.__ab[0], self.__ab[1], self.__ab[2]
        system = lambda x: np.array([
            C[0] * x[0] - B[1] * (u[1] * x[1] + v[1] * x[0] + w[1]) - self.__first_func(x[0]),
            -A[-2] * (u[-2] * x[1] + v[-2] * x[0] + w[-2]) + C[-1] * x[1] - self.__second_func(x[1])
        ], dtype=np.float64)
        prime = lambda x: np.array([
            [C[0] - B[1] * v[1] - self.__first_prime(x[0]), -B[1] * u[1]],
            [-A[-2] * v[-2], -A[-2] * u[-2] + C[-1] - self.__second_prime(x[1])]
        ], dtype=np.float64)
        y = fsolve(system, start_vector, fprime=prime)
        layer = v * y[0] + u * y[1] + w
        return layer

    def __subsystem(self):
        sub = np.row_stack((-self.__ab[0], self.__ab[1], -self.__ab[2]))
        sub[1, 0] = 1
        sub[0, 1] = 0
        sub[1, -1] = 1
        sub[2, -2] = 0
        return sub

    def __solve_subsystems(self):
        n = self.__ab.shape[1]
        ub = np.array([0] * (n - 1) + [1])
        wb = np.concatenate(([0], self.__b, [0]))
        vb = np.array([1] + [0] * (n - 1))
        system = self.__subsystem()
        solve = lambda b: solve_banded((1, 1), system, b)
        return solve(vb), solve(ub), solve(wb)

    def __solve(self, system, prime, start_vector):
        x0 = start_vector
        print(x0)
        print(prime(x0), system(x0))
        delta = np.linalg.solve(prime(x0), system(x0))
        x1 = x0 + delta
        while np.max(np.abs(x1 - x0)) >= 1e-10:
            print(x0)
            x0 = x1
            delta = np.linalg.solve(prime(x0), system(x0))
            x1 = x0 + delta
        return x1