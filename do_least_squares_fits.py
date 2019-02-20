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
parser.add_argument('--build_type', type=str, default='Release')
args = parser.parse_args()

################################################################################
# preliminaries

import sys;
sys.path.insert(0, '../build/%s' % args.build_type)
sys.path.insert(0, '../misc/py')

import common
import common3d
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyeikonal as eik
import time

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

Olims2 = [eik.Olim4Rect, eik.Olim8Rect]
Olims3 = [eik.Olim6Rect, eik.Olim18Rect, eik.Olim26Rect, eik.Olim3dHuRect]

################################################################################
# 2D

T2 = dict()

for Olim in Olims2:
    print(common.get_marcher_name(Olim))

    T2[Olim] = np.empty(len(N2))

    for i, n in enumerate(N2):
        print('- n %d (%d/%d)' % (n, i + 1, len(N2)))

        s = np.ones((n, n))
        h = 2/(n-1)
        i0 = n//2

        t = np.inf
        for _ in range(1 if n > 1000 else 5):
            olim = Olim(s, h)
            olim.add_boundary_node(i0, i0)

            t0 = time.perf_counter()
            olim.run()
            t = min(t, time.perf_counter() - t0)

        print('    - %s' % datetime.timedelta(seconds=t))

        T2[Olim][i] = t


################################################################################
# 3D

T3 = dict()

for Olim in Olims3:
    print(common3d.get_marcher_name(Olim))

    T3[Olim] = np.empty(len(N3))

    for i, n in enumerate(N3):
        print('- n %d (%d/%d)' % (n, i + 1, len(N3)))

        s = np.ones((n, n, n))
        h = 2/(n-1)
        i0 = n//2

        t = np.inf
        for _ in range(1 if n > 100 else 5):
            olim = Olim(s, h)
            olim.add_boundary_node(i0, i0, i0)

            t0 = time.perf_counter()
            olim.run()
            t = min(t, time.perf_counter() - t0)

        print('    - %s' % datetime.timedelta(seconds=t))

        T3[Olim][i] = t

################################################################################
# Compute alpha estimate

A2 = dict()
for Olim in Olims2:
    A2[Olim] = \
        (np.log2(T2[Olim][1:]) - np.log2(T2[Olim][:-1]))/(2*(P2[1:] - P2[:-1]))

A3 = dict()
for Olim in Olims3:
    A3[Olim] = \
        (np.log2(T3[Olim][1:]) - np.log2(T3[Olim][:-1]))/(3*(P3[1:] - P3[:-1]))
        
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
for i, Olim in enumerate(Olims2):
    ax.loglog(
        # N2[1:],
        N2,
        T2[Olim],
        label=common.get_marcher_plot_name(Olim),
        color=colors[i],
        **style)
ax.minorticks_off()
ax.set_xticks(N2[::3])
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in P2[::3]])
ax.set_xlabel('$N$')
ax.legend(loc='lower right')

ax = axes[1]
for i, Olim in enumerate(Olims3):
    ax.loglog(
        # N3[1:],
        N3,
        T3[Olim],
        label=common3d.get_marcher_plot_name(Olim),
        color=colors[i],
        **style)
ax.minorticks_off()
ax.set_xticks(N3[::2])
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in P3[::2]])
ax.set_xlabel('$N$')
ax.legend(loc='lower right')

fig.tight_layout()

fig.savefig('qv-time-plots.eps')

################################################################################
# Least squares fit

polyfit = np.polynomial.polynomial.polyfit

Alpha2 = {Olim: polyfit(P2, np.log2(T2[Olim]), 1) for Olim in Olims2}
Alpha3 = {Olim: polyfit(P3, np.log2(T3[Olim]), 1) for Olim in Olims3}

for Olim, (log2C, alpha) in Alpha2.items():
    C = 2**log2C
    print('%s: C = %0.4g, alpha = %0.4g' % (common.get_marcher_name(Olim), C, alpha))

for Olim, (log2C, alpha) in Alpha3.items():
    C = 2**log2C
    print('%s: C = %0.4g, alpha = %0.4g' % (common3d.get_marcher_name(Olim), C, alpha))
