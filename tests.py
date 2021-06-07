import numpy as np
from equationsolver import Equation
from equation import *
from plot import Plot
from typing import Tuple, Callable, List


class Test:
    def __init__(self,
                 method: SolutionMethods,
                 real: Callable[[float, float], float] = None):
        self.method = method
        self.real = real

    def draw(self,
             steps: Tuple[float, float],
             equation: Equation,
             name: str,
             epsilon: float = 1e-10,
             test_mode_enabled: bool = False):
        solution = equation.solve(steps, self.method, epsilon)
        if not test_mode_enabled:
            plot = Plot(equation)
            file = lambda type: f'Plots/{self.method.name.lower()}/{name}/{type}.png'
            plot.draw(steps, real_solution=self.real, title='Аналитическое решение',
                      save_to=file('real'))
            plot.draw(steps, values=solution, title='Численное решение',
                      save_to=file('numeric'))
            plot.draw(steps, 'Погрешность решения', 'error', solution, self.real,
                      save_to=file('error'))


class RealTest:
    def __init__(self, method: SolutionMethods):
        self.method = method

    def draw(self,
             steps: Tuple[float, float],
             equations: List[Equation],
             name: str,
             epsilon: float = 1e-10,
             test_mode_enabled: bool = False):
        for equation in equations:
            solution = equation.solve(steps, self.method, epsilon)
            if not test_mode_enabled:
                temperature = 100 * solution
                plot = Plot(equation)
                T = int(equation.sections.t[1])
                file = lambda time: f'Plots/{self.method.name.lower()}/' \
                                    f'{name}/{time}hours.png'
                plot.draw(steps, f'After {T} hour\'s', values=temperature,
                          save_to=file(T))


class TestConstant(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: 1.)

    def run(self,
            steps: Tuple[float, float] = (0.01, 0.01),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1., 1., 1., lambda x, t: 0.),
            BoundaryConditions(
                lambda x: 1.,
                lambda t: 1.,
                lambda t: 1.
            )
        )
        self.draw(steps, equation, 'constant', epsilon, test_mode_enabled)


class TestPlane(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: x + t)

    def run(self,
            steps: Tuple[float, float] = (0.01, 0.01),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1., 1., 1., lambda x, t: 1.),
            BoundaryConditions(
                lambda x: x,
                lambda t: np.power(t, 4) - 1,
                lambda t: np.power(t + 1., 4) + 1
            )
        )
        self.draw(steps, equation, 'plane', epsilon, test_mode_enabled)


class TestLinearQuadratic(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: t + np.square(x) / 2)

    def run(self,
            steps: Tuple[float, float] = (0.01, 0.01),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1., 1., 1., lambda x, t: 0.),
            BoundaryConditions(
                lambda x: np.square(x) / 2,
                lambda t: np.power(t, 4),
                lambda t: 1. + np.power(0.5 + t, 4)
            )
        )
        self.draw(steps, equation, 'linear_quadratic', epsilon, test_mode_enabled)


class TestX2(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: np.square(x))

    def run(self,
            steps: Tuple[float, float] = (0.01, 0.01),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1., 1., 1., lambda x, t: -2.),
            BoundaryConditions(
                lambda x: np.square(x),
                lambda t: 0.,
                lambda t: 3.
            )
        )
        self.draw(steps, equation, 'quadratic', epsilon, test_mode_enabled)


class TestXT2(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: x * np.square(t) / 2)

    def run(self,
            steps: Tuple[float, float] = (0.01, 0.001),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1, 1, 1, lambda x, t: x * t),
            BoundaryConditions(
                lambda x: 0.,
                lambda t: -np.square(t) / 2,
                lambda t: np.square(t) / 2 + np.power(np.square(t) / 2, 4)
            )
        )
        self.draw(steps, equation, 'hyperbolic', epsilon, test_mode_enabled)


class TestX3(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: np.power(x, 3))

    def run(self,
            steps: Tuple[float, float] = (0.025, 0.001),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1., 1., 1., lambda x, t: -6.0 * x),
            BoundaryConditions(
                lambda x: np.power(x, 3),
                lambda t: 0.,
                lambda t: 4.
            )
        )
        self.draw(steps, equation, 'cube', epsilon, test_mode_enabled)


class TestT2LnX(Test):
    def __init__(self, method: SolutionMethods):
        super().__init__(method, lambda x, t: np.square(t) * np.log(x + 1))

    def run(self,
            steps: Tuple[float, float] = (0.025, 0.0025),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equation = Equation(
            Sections((0., 1.), (0., 1.)),
            Parameters(1, 1, 1, lambda x, t: np.square(t / (x + 1)) + 2 * t * np.log(x + 1)),
            BoundaryConditions(
                lambda x: 0.,
                lambda t: -np.square(t),
                lambda t: np.square(t) / 2 + np.power(np.square(t) * np.log(2), 4)
            )
        )
        self.draw(steps, equation, 'logarithmic', epsilon, test_mode_enabled)


class RealCoolingTest(RealTest):
    def __init__(self, method: SolutionMethods):
        super().__init__(method)

    def run(self,
            steps: Tuple[float, float] = (0.025, 0.005),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equations = [Equation(
            Sections((0., 1.), (0., float(T))),
            Parameters(40, 1e4 / 9, 5.67, lambda x, t: 0),
            BoundaryConditions(
                lambda x: 5.73,
                lambda t: 5.67 * np.power(2.73, 4),
                lambda t: 5.67 * np.power(2.73, 4)
            )
        ) for T in [1, 6, 12, 24]]
        self.draw(steps, equations, 'cooling', epsilon, test_mode_enabled)


class RealHeatingTest(RealTest):
    def __init__(self, method: SolutionMethods):
        super().__init__(method)

    def run(self,
            steps: Tuple[float, float] = (0.025, 0.005),
            epsilon: float = 1e-10,
            test_mode_enabled: bool = False):
        equations = [Equation(
            Sections((0., 1.), (0., float(T))),
            Parameters(40, 1e4 / 9, 5.67, lambda x, t: 0),
            BoundaryConditions(
                lambda x: 2.73,
                lambda t: 5.67 * np.power(5.73, 4),
                lambda t: 5.67 * np.power(5.73, 4)
            )
        ) for T in [1, 6, 12, 24]]
        self.draw(steps, equations, 'heating', epsilon, test_mode_enabled)


tests = [TestConstant, TestPlane, TestX2,
         TestLinearQuadratic, TestX3, TestXT2,
         TestT2LnX,
         RealCoolingTest, RealHeatingTest]
