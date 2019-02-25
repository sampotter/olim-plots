#!/usr/bin/env python3

################################################################################
# parse arguments first

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_3d_power', type=int, default=3)
parser.add_argument('--max_3d_power', type=int, default=8)
parser.add_argument('--no_factoring', action='store_true')
args = parser.parse_args()

################################################################################
# preliminaries

import common3
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pyolim
import slow3

from pyolim import Neighborhood, Quadrature

from cycler import cycler
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

Npows = np.arange(args.min_3d_power, args.max_3d_power + 1)
N = 2**Npows + 1

r_fac = 0.1

Slows = [slow3.s1, slow3.s2, slow3.s3, slow3.s4]
Solns = {slow3.s1: slow3.f1, slow3.s2: slow3.f2, slow3.s3: slow3.f3, slow3.s4: slow3.f4}

Nbs = [Neighborhood.OLIM6, Neighborhood.OLIM18, Neighborhood.OLIM26, Neighborhood.OLIM3D]
Quads = [Quadrature.MP0, Quadrature.MP1, Quadrature.RHR]

Slow_by_Nb_by_Quad = list(itertools.product(Slows, Nbs, Quads))
E = {(slow, nb, quad): np.empty(N.shape) for slow, nb, quad in Slow_by_Nb_by_Quad}

current_slow = None
current_nb = None
current_quad = None
current_n = None

for (slow, nb, quad), (ind, n) in itertools.product(Slow_by_Nb_by_Quad, enumerate(N)):
    if slow != current_slow:
        print(slow3.get_slowness_func_name(slow))
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

    h = 2/(n-1)
    i0, j0, k0 = n//2, n//2, n//2
    L = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(L, L, L)
    R = np.sqrt(x**2 + y**2 + z**2)
    I, J, K = np.where(R < r_fac)
    u, S = slow3.get_fields(Solns[slow], slow, x, y, z)

    olim = pyolim.Olim(nb, quad, S, h)
    if not args.no_factoring:
        fac_src = pyolim.FacSrc((i0, j0, k0), slow(0, 0, 0))
        for i, j, k in zip(I, J, K):
            olim.set_fac_src((i, j, k), fac_src)
    olim.add_src((i0, j0, k0))
    olim.run()

    E[slow, nb, quad][ind] = \
        norm((u - olim.U).flatten(), np.inf)/norm(u.flatten(), np.inf)

################################################################################
# plotting

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cmap = [0, 1, 4, 3]
linestyles = [':', '-.', '--', '-']

style = {
    'linewidth': 1,
    'marker': '|',
    'markersize': 3.5
}

fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(6.5, 6.5))
ax = axes.flatten()

for i, slow in enumerate(Slows):
    for ind, (nb, quad) in enumerate(itertools.product(Nbs, Quads)):
        print((i, ind, nb, quad))
        ax[i].loglog(
            N, E[slow, nb, quad], color=colors[cmap[ind % 3]],
            linestyle=linestyles[ind//3],
            label=common3.get_marcher_plot_name(nb, quad), **style)
        ax[i].text(
            0.95, 0.9, '$\\texttt{s%d}$' % (i + 1),
            transform=ax[i].transAxes, horizontalalignment='center',
            verticalalignment='center')

axes[0, 0].set_ylabel(r'$\|u - U\|_\infty/\|u\|_\infty$')
axes[1, 0].set_ylabel(r'$\|u - U\|_\infty/\|u\|_\infty$')
axes[-1, 0].set_xlabel(r'$N$')
axes[-1, 1].set_xlabel(r'$N$')

handles, labels = axes[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
fig.tight_layout()
fig.subplots_adjust(0.085, 0.055, 0.995, 0.935) # TODO: adjust this
fig.savefig('size_vs_error_3d.eps')
