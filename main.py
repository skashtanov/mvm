from tests import *
from time_measure import ExecutionTime
from warnings import filterwarnings

if __name__ == '__main__':
    filterwarnings('ignore')
    amounts = 20
    average = dict.fromkeys(SolutionMethods)
    for method in SolutionMethods:
        total = 0.0
        for i in range(amounts):
            for test in tests:
                result, measure = ExecutionTime(test(method).run, {'test_mode_enabled': True}).time()
                total += measure
        average[method] = total / amounts
    print(average)
