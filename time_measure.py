from time import time
from typing import Callable, List


class ExecutionTime:
    def __init__(self, function: Callable, *args, **kwargs):
        self.__function = function
        self.__args = args
        self.__kwargs = kwargs

    def time(self):
        start = time()
        result = self.__function(**self.__kwargs)
        finish = time()
        return result, finish - start
