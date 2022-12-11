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


def multivar_func_classic(x):
    return x[0] * x[0] + 10 * math.sin(x[1]) + 2 * math.cos(x[2])


def run_parallel_func(eval_func, param_values, Y):
    for i in numba.prange(len(param_values)):
        Y[i] = eval_func(param_values[i])

    return None


if __name__ == "__main__":
    param_values = saltelli.sample(problem, N_SAMPLES)
    Y = np.zeros([param_values.shape[0]])
    start = time.time()
    run_parallel_func(multivar_func_classic, param_values, Y)
    eval_time = round(time.time() - start, 2)
    print(eval_time)
