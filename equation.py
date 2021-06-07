from enum import Enum
from typing import Callable, Tuple

class SolutionMethods(Enum):
    NEWTON = 1,
    PARAMETRIZED = 2


class Sections:
    def __init__(self,
                 x_section: Tuple[float, float],
                 t_section: Tuple[float, float]):
        self.x = x_section
        self.t = t_section


class Parameters:
    def __init__(self,
                 k: float, cp: float, sigma: float,
                 function: Callable[[float, float], float]):
        self.k = k
        self.cp = cp
        self.sigma = sigma
        self.function = function


class BoundaryConditions:
    def __init__(self,
                 initial: Callable[[float], float],
                 left : Callable[[float], float],
                 right: Callable[[float], float]):
        self.initial = initial
        self.left = left
        self.right = right
