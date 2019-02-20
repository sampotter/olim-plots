#!/usr/bin/env python3


################################################################################
# parse arguments first

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_3d_power', type=int, default=3)
parser.add_argument('--max_3d_power', type=int, default=8)
parser.add_argument('--build_type', type=str, default='Release')
parser.add_argument('--no_factoring', action='store_true')
args = parser.parse_args()

################################################################################
# preliminaries

import sys
sys.path.insert(0, '../build/%s' % args.build_type)
sys.path.insert(0, '../misc/py')

import common3d
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pyeikonal as eik
import speedfuncs3d

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

Slows = [speedfuncs3d.s1, speedfuncs3d.s2, speedfuncs3d.s3, speedfuncs3d.s4]
Solns = {
    speedfuncs3d.s1: speedfuncs3d.f1,
    speedfuncs3d.s2: speedfuncs3d.f2,
    speedfuncs3d.s3: speedfuncs3d.f3,
    speedfuncs3d.s4: speedfuncs3d.f4
}
Olims = [eik.Olim6Mid0, eik.Olim6Mid1, eik.Olim6Rect,
         eik.Olim18Mid0, eik.Olim18Mid1, eik.Olim18Rect,
         eik.Olim26Mid0, eik.Olim26Mid1, eik.Olim26Rect,
         eik.Olim3dHuMid0, eik.Olim3dHuMid1, eik.Olim3dHuRect]

Slows_by_Olims = list(itertools.product(Slows, Olims))

E = {(slow, Olim): np.empty(N.shape) for slow, Olim in Slows_by_Olims}

current_slow, current_Olim, current_n = None, None, None
for (slow, Olim), (ind, n) in itertools.product(Slows_by_Olims, enumerate(N)):
    if slow != current_slow:
        print(speedfuncs3d.get_slowness_func_name(slow))
        current_slow = slow
    if Olim != current_Olim:
        print('* %s' % str(Olim))
        current_Olim = Olim
    if n != current_n:
        print('  - %d' % n)
        current_n = n

    h = 2/(n-1)
    i0, j0, k0 = n//2, n//2, n//2
    L = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(L, L, L)
    R = np.sqrt(x**2 + y**2 + z**2)
    I, J, K = np.where(R < r_fac)
    u, S = speedfuncs3d.get_fields(Solns[slow], slow, x, y, z)

    o = Olim(S, h)
    if not args.no_factoring:
        fc = eik.FacCenter3d(i0, j0, k0, slow(0, 0, 0))
        for i, j, k in zip(I, J, K):
            o.set_node_fac_center(i, j, k, fc)
    o.add_boundary_node(i0, j0, k0)
    o.run()

    U = np.array([[[o.get_value(i, j, k) for k in range(n)]
                   for j in range(n)]
                  for i in range(n)])
    E[slow, Olim][ind] = \
        norm((u - U).flatten(), np.inf)/norm(u.flatten(), np.inf)

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
    for ind, Olim in enumerate(Olims):
        print((i, ind, Olim))
        ax[i].loglog(
            N, E[slow, Olim], color=colors[cmap[ind % 3]],
            linestyle=linestyles[ind//3],
            label=common3d.get_marcher_plot_name(Olim), **style)
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
fig.savefig('make_3d_error_plots.eps')
fig.show()
