#!/usr/bin/env python3

################################################################################
# parse arguments first

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_2d_power', type=int, default=3)
parser.add_argument('--max_2d_power', type=int, default=15)
args = parser.parse_args()

################################################################################
# preliminaries

import common
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pyolim
import slow2
import time

from pyolim import Neighborhood
from pyolim import Quadrature

from matplotlib import rc

rc('text', usetex=True)
rc('font', **{
    'family': 'serif',
    'serif': ['Computer Modern'],
    'size': 8
})

norm = np.linalg.norm

plt.ion()
plt.style.use('bmh')

Npows = np.arange(args.min_2d_power, args.max_2d_power + 1)
N = 2**Npows + 1

use_local_factoring = True
r_fac = 0.1

Slows = [slow2.s1, slow2.s2, slow2.s3, slow2.s4]
Solns = {
    slow2.s1: slow2.f1,
    slow2.s2: slow2.f2,
    slow2.s3: slow2.f3,
    slow2.s4: slow2.f4
}

Nbs = [Neighborhood.OLIM4, Neighborhood.OLIM8]
Quads = [Quadrature.MP0, Quadrature.MP1, Quadrature.RHR]

Slow_x_Nb_x_Quad = list(itertools.product(Slows, Nbs, Quads))

T = {(slow, nb, quad): np.empty(N.shape) for slow, nb, quad in Slow_x_Nb_x_Quad}
E2 = {(slow, nb, quad): np.empty(N.shape) for slow, nb, quad in Slow_x_Nb_x_Quad}
EI = {(slow, nb, quad): np.empty(N.shape) for slow, nb, quad in Slow_x_Nb_x_Quad}

ntrials = 2

current_slow = None
current_nb = None
current_quad = None
current_n = None

for (slow, nb, quad), (ind, n) in itertools.product(Slow_x_Nb_x_Quad, enumerate(N)):
    if slow != current_slow:
        print(slow2.get_slowness_func_name(slow))
        current_slow = slow
    if nb != current_nb or quad != current_quad:
        print('* %s %s' % (str(nb), str(quad)))
    if nb != current_nb:
        current_nb = nb
    if quad != current_quad:
        current_quad = quad
    if n != current_n:
        print('  - %d' % n)
        current_n = n

    # get timings

    h = 2/(n-1)
    i0, j0 = n//2, n//2
    L = np.linspace(-1, 1, n)
    x, y = np.meshgrid(L, L)
    R = np.sqrt(x**2 + y**2)
    I, J = np.where(R < r_fac)
    u, S = slow2.get_fields(Solns[slow], slow, x, y)

    t = np.inf

    for _ in range(ntrials):

        olim = pyolim.Olim(nb, quad, S, h)
        if use_local_factoring:
            fac_src = pyolim.FacSrc((i0, j0), slow(0, 0))
            for i, j in zip(I, J):
                olim.set_fac_src((i, j), fac_src)
        olim.add_src((i0, j0))

        t0 = time.perf_counter()
        olim.run()
        t = min(t, time.perf_counter() - t0)

    T[slow, nb, quad][ind] = t

    # get errors

    diff = u - olim.U
    E2[slow, nb, quad][ind] = norm(diff, 'fro')/norm(u, 'fro')
    EI[slow, nb, quad][ind] = norm(diff, np.inf)/norm(u, np.inf)

# make plots
 
marker = '*'
linestyles = ['solid', 'dashed', 'dotted']
colors = ['#1f77b4', '#d62728']

# time vs error

fig, axes = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(6.5, 5.5))

axes[0, 0].set_title('Relative $\ell_2$ Error')
axes[0, 1].set_title('Relative $\ell_\infty$ Error')

for row, slow in enumerate(Slows):
    for ind, (nb, quad) in enumerate(itertools.product(Nbs, Quads)):
        name = common.get_marcher_plot_name(nb, quad)
        axes[row, 0].loglog(
            T[slow, nb, quad], E2[slow, nb, quad], marker=marker, color=colors[ind//3],
            linestyle=linestyles[ind % 3], linewidth=1, label=name)
        axes[row, 0].text(0.95, 0.9, '$\\texttt{s%d}$' % (row + 1),
                          transform=axes[row, 0].transAxes,
                          horizontalalignment='center',
                          verticalalignment='center')
        axes[row, 1].loglog(
            T[slow, nb, quad], EI[slow, nb, quad], marker=marker, color=colors[ind//3],
            linestyle=linestyles[ind % 3], linewidth=1, label=name)
        axes[row, 1].text(0.95, 0.9, '$\\texttt{s%d}$' % (row + 1),
                          transform=axes[row, 1].transAxes,
                          horizontalalignment='center',
                          verticalalignment='center')

axes[-1, 0].set_xlabel('Time (s.)')    
axes[-1, 1].set_xlabel('Time (s.)')    

handles, labels = axes[-1, -1].get_legend_handles_labels()
    
fig.legend(handles, labels, loc='upper center', ncol=3)
fig.tight_layout()
fig.subplots_adjust(0.05, 0.075, 0.995, 0.8625)
fig.savefig('time_vs_error_2d.eps')
fig.show()

# size vs error

fig, axes = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(6.5, 5.5))

axes[0, 0].set_title('Relative $\ell_2$ Error')
axes[0, 1].set_title('Relative $\ell_\infty$ Error')

for row, slow in enumerate(Slows):
    for ind, (nb, quad) in enumerate(itertools.product(Nbs, Quads)):
        name = common.get_marcher_plot_name(nb, quad)
        axes[row, 0].loglog(
            N, E2[slow, nb, quad], marker=marker, color=colors[ind//3],
            linestyle=linestyles[ind % 3], linewidth=1, label=name)
        axes[row, 0].text(0.95, 0.9, '$\\texttt{s%d}$' % (row + 1),
                          transform=axes[row, 0].transAxes,
                          horizontalalignment='center',
                          verticalalignment='center')
        axes[row, 0].minorticks_off()
        axes[row, 1].loglog(
            N, EI[slow, nb, quad], marker=marker, color=colors[ind//3],
            linestyle=linestyles[ind % 3], linewidth=1, label=name)
        axes[row, 1].text(0.95, 0.9, '$\\texttt{s%d}$' % (row + 1),
                          transform=axes[row, 1].transAxes,
                          horizontalalignment='center',
                          verticalalignment='center')
        axes[row, 1].minorticks_off()

axes[-1, 0].set_xlabel('$N$')

xticklabels = ['$2^{%d} + 1$' % p for p in Npows]
axes[-1, 1].set_xlabel('$N$')    
axes[-1, 1].set_xticks(N[::2])
axes[-1, 1].set_xticklabels(xticklabels[::2])

handles, labels = axes[-1, -1].get_legend_handles_labels()
    
fig.legend(handles, labels, loc='upper center', ncol=3)
fig.tight_layout()
fig.subplots_adjust(0.05, 0.075, 0.995, 0.8625)
fig.savefig('size_vs_error_2d.eps')
fig.show()
