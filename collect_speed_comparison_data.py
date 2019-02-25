#!/usr/bin/env python3

################################################################################
# preliminaries

import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pyolim
import time

from pyolim import Neighborhood
from pyolim import Quadrature

from matplotlib import rc

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

plt.ion()
plt.style.use('bmh')

def tic():
    global t0
    t0 = time.perf_counter()

def toc():
    global t0
    return time.perf_counter() - t0

################################################################################
# gather timings

N = np.concatenate([
    np.logspace(3, 6, 12, base=2, dtype=int, endpoint=False),
    np.logspace(6, 9, 10, base=2, dtype=int)])

Tb, To = [], []

for i, n in enumerate(N):

    h = 2/(n-1)
    i0 = n//2
    S = np.ones((n, n, n))

    tb, to = np.inf, np.inf

    ntrials = 10 if n < 120 else 3

    for _ in range(ntrials):

        olim = pyolim.Olim(Neighborhood.FMM3, Quadrature.RHR, S, h)
        olim.add_src((i0, i0, i0))

        tic()
        olim.run()
        tb = min(tb, toc())

        olim = pyolim.Olim(Neighborhood.OLIM6, Quadrature.RHR, S, h)
        olim.add_src((i0, i0, i0))

        tic()
        olim.run()
        to = min(to, toc())

    Tb.append(tb)
    To.append(to)

    print('n = %d, tb = %g, to = %g, to/tb = %g' % (n, tb, to, to/tb))

Tb, To = np.array(Tb), np.array(To)

np.savez('speed_comparison.npz', N=N, Tb=Tb, To=To)
