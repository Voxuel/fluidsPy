from functools import reduce
from itertools import cycle
from math import factorial

import numpy as np
import scipy.sparse as sp

def difference(derivative, accuracy=1):
    derivative += 1
    radius = accuracy + derivative // 2 - 1
    points = range(-radius, radius + 1)
    coefficients = np.linalg.inv(np.vander(points))
    return coefficients[-derivative] * factorial(derivative - 1), points


def operator(shape, *differance):
    differance = zip(shape, cycle(differance))
    factors = (sp.diags(*diff, shape=(dim,) * 2) for dim, diff in differance)
    return reduce(lambda a, f: sp.kronsum(f, a, format='csc'), factors)