#!/usr/bin/env python3

################################################################################
# parse arguments first

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_2d_power', type=int, default=3)
parser.add_argument('--max_2d_power', type=int, default=15)
parser.add_argument('--min_3d_power', type=int, default=3)
parser.add_argument('--max_3d_power', type=int, default=10)
parser.add_argument('--steps', type=int, default=1)
args = parser.parse_args()

################################################################################
# preliminaries

import common2
import common3
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyolim
import time

from pyolim import Neighborhood, Quadrature

plt.ion()

from matplotlib.colors import LogNorm
from numpy.linalg import norm

plt.rc('text', usetex=True)
plt.rc('font', **{
    'family': 'serif',
    'serif': ['Computer Modern'],
    'size': 8
})

plt.style.use('bmh')

################################################################################
# parameters

P2 = np.linspace(
    args.min_2d_power,
    args.max_2d_power,
    args.steps*(args.max_2d_power - args.min_2d_power) + 1)
# P2 = np.arange(args.min_2d_power, args.max_2d_power + 1)
N2 = np.round(2**P2 + 1).astype(int)

P3 = np.linspace(
    args.min_3d_power,
    args.max_3d_power,
    args.steps*(args.max_3d_power - args.min_3d_power) + 1)
# P3 = np.arange(args.min_3d_power, args.max_3d_power + 1)
N3 = np.round(2**P3 + 1).astype(int)

quad = Quadrature.RHR
Nb2 = [Neighborhood.OLIM4, Neighborhood.OLIM8]
Nb3 = [Neighborhood.OLIM6, Neighborhood.OLIM18, Neighborhood.OLIM26, Neighborhood.OLIM3D]

################################################################################
# 2D

T2 = dict()

for nb in Nb2:
    print('%s, %s' % (nb, quad))

    T2[nb] = np.empty(len(N2))

    for i, n in enumerate(N2):
        print('- n %d (%d/%d)' % (n, i + 1, len(N2)))

        s = np.ones((n, n))
        h = 2/(n-1)
        i0 = n//2

        t = np.inf
        for _ in range(1 if n > 1000 else 5):
            olim = pyolim.Olim(nb, quad, s, h)
            olim.add_src((i0, i0))

            t0 = time.perf_counter()
            olim.run()
            t = min(t, time.perf_counter() - t0)

        print('    - %s' % datetime.timedelta(seconds=t))

        T2[nb][i] = t


################################################################################
# 3D

T3 = dict()

for nb in Nb3:
    print('%s, %s' % (nb, quad))

    T3[nb] = np.empty(len(N3))

    for i, n in enumerate(N3):
        print('- n %d (%d/%d)' % (n, i + 1, len(N3)))

        s = np.ones((n, n, n))
        h = 2/(n-1)
        i0 = n//2

        t = np.inf
        for _ in range(1 if n > 100 else 5):
            olim = pyolim.Olim(nb, quad, s, h)
            olim.add_src((i0, i0, i0))

            t0 = time.perf_counter()
            olim.run()
            t = min(t, time.perf_counter() - t0)

        print('    - %s' % datetime.timedelta(seconds=t))

        T3[nb][i] = t

################################################################################
# Compute alpha estimate

def alpha_est(T, P, n):
    return (np.log2(T[1:]) - np.log2(T[:-1]))/(n*(P[1:] - P[:-1]))

A2 = {nb: alpha_est(T2[nb], P2, 2) for nb in Nb2}
A3 = {nb: alpha_est(T3[nb], P3, 3) for nb in Nb3}
        
################################################################################
# Plotting

style = {
    'linestyle': 'solid',
    'linewidth': 1,
    'marker': '|',
    'markersize': 3.5
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(1, 2, sharey='all', figsize=(6.5, 2))

axes[0].set_ylabel(r'Time (s.)')

ax = axes[0]
for i, nb in enumerate(Nb2):
    ax.loglog(N2, T2[nb], label=common2.get_marcher_plot_name(nb),
              color=colors[i], **style)
ax.minorticks_off()
ax.set_xticks(N2[::3])
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in P2[::3]])
ax.set_xlabel('$N$')
ax.legend(loc='lower right')

ax = axes[1]
for i, nb in enumerate(Nb3):
    ax.loglog(N3, T3[nb], label=common3.get_marcher_plot_name(nb),
              color=colors[i], **style)
ax.minorticks_off()
ax.set_xticks(N3[::2])
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in P3[::2]])
ax.set_xlabel('$N$')
ax.legend(loc='lower right')

fig.tight_layout()

fig.savefig('qv_time_plots.eps')

################################################################################
# Least squares fit

polyfit = np.polynomial.polynomial.polyfit

Alpha2 = {nb: polyfit(P2, np.log2(T2[nb]), 1) for nb in Nb2}
Alpha3 = {nb: polyfit(P3, np.log2(T3[nb]), 1) for nb in Nb3}

for nb, (log2C, alpha) in Alpha2.items():
    C = 2**log2C
    print('%s, %s: C = %0.4g, alpha = %0.4g' % (nb, quad, C, alpha))

for nb, (log2C, alpha) in Alpha3.items():
    C = 2**log2C
    print('%s, %s: C = %0.4g, alpha = %0.4g' % (nb, quad, C, alpha))
