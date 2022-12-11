import codecs
import json
import math
import time

import numpy as np
import numba
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

N_SAMPLES = 2**20

problem = {
    'num_vars': 3,
    'names': ['first' , 'second', 'third'],
    'bounds': [[-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]]
}

def run_sobol_analysis(problem, eval_func, parallel_SA=False):
    """
    Функция для тестирования функции нескольких переменных
    problem: кофиг параметров
    eval_func: функция нескольких переменных
    parallel_SA: флаг sobol.analyze
    return: sobol_ - коэффиценты анализа Соболя, sampling_time - время на генерацию данных, eval_time - время выполнения функции, sa_time - время на анализ
    """
    start = time.time()
    param_values = saltelli.sample(problem, N_SAMPLES)
    sampling_time = round(time.time() - start, 2)

    Y = np.zeros([param_values.shape[0]])

    start = time.time()
    for i in numba.prange(len(param_values)):
        Y[i] = eval_func(param_values[i])
    eval_time = round(time.time() - start, 2)

    start = time.time()
    sobol_ = sobol.analyze(problem, Y, parallel=parallel_SA) #, print_to_console=True
    sa_time = round(time.time() - start, 2)

    return sobol_, sampling_time, eval_time, sa_time


def multivar_func_classic(x):
    return x[0] * x[0] + 10 * math.sin(x[1]) + 2 * math.cos(x[2])


@numba.njit
def multivar_func_numba(x):
    return x[0] * x[0] + 10 * math.sin(x[1]) + 2 * math.cos(x[2])



if __name__ == '__main__':
    # Si, sample_time, eval_time, sa_time = run_sobol_analysis(problem, multivar_func_classic)
    Si, sample_time, eval_time, sa_time = run_sobol_analysis(problem, multivar_func_classic)
    Si_filter = {k:Si[k] for k in ['ST','ST_conf','S1','S1_conf']}
    Si_df = pd.DataFrame(Si_filter, index=problem['names'])
    Si_df.to_csv('./result_dump/parallel_result.csv')
    print(Si_df)
    print(sample_time)
    print(eval_time) 
    print(sa_time)
