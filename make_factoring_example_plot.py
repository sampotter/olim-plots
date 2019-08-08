#!/usr/bin/env python3

################################################################################
# parse arguments first

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_2d_power', type=int, default=3)
parser.add_argument('--max_2d_power', type=int, default=15)
parser.add_argument('--min_3d_power', type=int, default=3)
parser.add_argument('--max_3d_power', type=int, default=10)
args = parser.parse_args()

################################################################################
# preliminaries

import matplotlib.pyplot as plt
import numpy as np
import pyolim
import slow2
import slow3

from pyolim import Neighborhood, Quadrature

from matplotlib.colors import LogNorm
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

nb2 = Neighborhood.OLIM8
nb3 = Neighborhood.OLIM26
quad = Quadrature.RHR

Npow = np.arange(args.min_2d_power, args.max_2d_power + 1)
N = 2**Npow + 1

Npow_3d = np.arange(args.min_3d_power, args.max_3d_power + 1)
N_3d = 2**Npow_3d + 1

rfacs = [0.05, 0.1, 0.15, 0.2]
nrfac = len(rfacs)

EI, EIfac = np.empty(len(N)), np.empty((len(N), nrfac))
EI_3d, EIfac_3d = np.empty(len(N_3d)), np.empty((len(N_3d), nrfac))

print('solving 2d problems')
for ind, n in enumerate(N):
    print('- n = %d' % n)

    h = 2/(n - 1)
    i0, j0 = n//2, n//2
    l = np.linspace(-1, 1, n)
    x, y = np.meshgrid(l, l)

    u, s = slow2.get_fields(slow2.f0, slow2.s0, x, y)

    # unfactored
    
    olim = pyolim.Olim(nb2, quad, s, h)
    olim.add_src((i0, j0))
    olim.run()
    EI[ind] = norm(u - olim.U, np.inf)/norm(u, np.inf)

    # factored using constant radius disk

    for rfac_ind, r_fac in enumerate(rfacs):
        print('  * r_fac = %g' % r_fac)
    
        R = np.sqrt(x**2 + y**2)
        I, J = np.where(R < r_fac)

        olim_fac = pyolim.Olim(nb2, quad, s, h)
        fac_src = pyolim.FacSrc((i0, j0), 1)
        for i, j in zip(I, J):
            olim_fac.set_fac_src((i, j), fac_src)
        olim_fac.add_src((i0, j0))
        olim_fac.run()
        EIfac[ind, rfac_ind] = norm(u - olim_fac.U, np.inf)/norm(u, np.inf)

print('- solving 3d problems')
for ind, n in enumerate(N_3d):
    print('- n = %d' % n)

    h = 2/(n - 1)
    i0, j0, k0 = n//2, n//2, n//2
    l = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(l, l, l)

    u, s = slow3.get_fields(slow3.f0, slow3.s0, x, y, z)

    # unfactored
    
    olim = pyolim.Olim(nb3, quad, s, h)
    olim.add_src((i0, j0, k0))
    olim.run()
    EI_3d[ind] = norm((u - olim.U).flatten(), np.inf)/norm(u.flatten(), np.inf)

    # factored using constant radius disk
    
    for rfac_ind, r_fac in enumerate(rfacs):
        print('  * r_fac = %g' % r_fac)

        R = np.sqrt(x**2 + y**2 + z**2)
        I, J, K = np.where(R < r_fac)

        print(I.shape)

        olim_fac = pyolim.Olim(nb3, quad, s, h)
        fac_src = pyolim.FacSrc((i0, j0, k0), 1)
        for i, j, k in zip(I, J, K):
            olim_fac.set_fac_src((i, j, k), fac_src)
        olim_fac.add_src((i0, j0, k0))
        olim_fac.run()
        EIfac_3d[ind, rfac_ind] = \
            norm((u - olim_fac.U).flatten(), np.inf)/norm(u.flatten(), np.inf)

################################################################################
# Plotting

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cmap = [0, 1, 4, 3]

fig, axes = plt.subplots(1, 2, sharey='row', figsize=(6.5, 3))
title_fontsize = 10

style = {
    'marker': '|',
    'markersize': 3.5,
    'linewidth': 1,
    'linestyle': '-'
}

axes[0].set_ylabel(r'$\|u - U\|_\infty/\|u\|_\infty$')

ax = axes[0]
tol = 1e-15
mask = EI > tol
ax.loglog(N[mask], EI[mask], label='Unfactored', color='k', **style)
for j, r_fac in enumerate(rfacs):
    mask = EIfac[:, j] > tol
    ax.loglog(N[mask], EIfac[mask, j], '-', color=colors[cmap[j]],
              label='Disk ($r_{fac} = %g$)' % r_fac, **style)
ax.set_title(r'\texttt{olim8rhr}', fontsize=title_fontsize)
ax.set_xlabel('$N$')
ax.set_xticks(N[::3])
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in Npow[::3]])

ax = axes[1]
tol = 1e-15
mask = EI_3d > tol
ax.loglog(N_3d[mask], EI_3d[mask], label='Unfactored', color='k', **style)
for j, r_fac in enumerate(rfacs):
    mask = EIfac_3d[:, j] > tol
    ax.loglog(N_3d[mask], EIfac_3d[mask, j], color=colors[cmap[j]],
              label='Disk ($r_{fac} = %g$)' % r_fac, **style)
ax.set_title(r'\texttt{olim26rhr}', fontsize=title_fontsize)
ax.set_xlabel('$N$')
ax.set_xticks(N_3d[::2])
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in Npow_3d[::2]])
ax.legend()

fig.tight_layout()

fig.savefig('factoring_example.eps')
