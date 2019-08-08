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

import common2
import common3
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyolim
import time

from pyolim import Neighborhood, Quadrature

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

P2 = np.arange(args.min_2d_power, args.max_2d_power + 1)
N = 2**P2 + 1
P3 = np.arange(args.min_3d_power, args.max_3d_power + 1)
N3D = 2**P3 + 1

R_fac = 0.1

vx, vy, vz = 5, 13, 20

x_fac_1, y_fac_1, z_fac_1 = 0.0, 0.0, 0.0
x_fac_2, y_fac_2, z_fac_2 = 0.8, 0.0, 0.0

Olims2 = [
    (Neighborhood.OLIM8, Quadrature.MP0),
    (Neighborhood.OLIM8, Quadrature.MP1),
    (Neighborhood.OLIM8, Quadrature.RHR),
]

Olims3 = [
    (Neighborhood.OLIM6, Quadrature.RHR),
    (Neighborhood.OLIM26, Quadrature.MP0),
    (Neighborhood.OLIM26, Quadrature.MP1),
    (Neighborhood.OLIM26, Quadrature.RHR),
    (Neighborhood.OLIM3D, Quadrature.MP0),
    (Neighborhood.OLIM3D, Quadrature.MP1),
    (Neighborhood.OLIM3D, Quadrature.RHR),
]

################################################################################
# 2D

s = lambda x, y: 1/(2 + vx*x + vy*y)
s_1, s_2 = s(x_fac_1, y_fac_1), s(x_fac_2, y_fac_2)

def make_u(x_fac, y_fac, vx, vy, s):
    return lambda x, y: \
        (1/np.sqrt(vx**2 + vy**2)) * \
        np.arccosh(
            1 +
            s(x_fac, y_fac)*s(x, y)*(vx**2 + vy**2)*(
                (x - x_fac)**2 + (y - y_fac)**2)/2)

u_1 = make_u(x_fac_1, y_fac_1, vx, vy, s)
u_2 = make_u(x_fac_2, y_fac_2, vx, vy, s)

u = lambda x, y: np.minimum(u_1(x, y), u_2(x, y))

E2 = dict()

for nb, quad in Olims2:
    print('%s %s' % (nb, quad))

    E2[nb, quad] = np.zeros(len(N))

    for k, n in enumerate(N):
        print('- n = %d (%d/%d)' % (n, k + 1, len(N)))

        L = np.linspace(0, 1, n)
        X, Y = np.meshgrid(L, L)
        u_ = u(X, Y)
        S = s(X, Y)

        h = 1/(n - 1)
        i_1, j_1 = y_fac_1/h, x_fac_1/h
        i_2, j_2 = y_fac_2/h, x_fac_2/h

        olim = pyolim.Olim(nb, quad, S, h)

        R_1 = np.sqrt((x_fac_1 - X)**2 + (y_fac_1 - Y)**2)
        fs_1 = pyolim.FacSrc((i_1, j_1), s_1)
        for i, j in zip(*np.where(R_1 <= R_fac)):
            olim.set_fac_src((i, j), fs_1)
        olim.add_src((x_fac_1, y_fac_1), s_1)

        R_2 = np.sqrt((x_fac_2 - X)**2 + (y_fac_2 - Y)**2)
        fs_2 = pyolim.FacSrc((i_2, j_2), s_2)
        for i, j in zip(*np.where(R_2 <= R_fac)):
            olim.set_fac_src((i, j), fs_2)
        olim.add_src((x_fac_2, y_fac_2), s_2)
        
        olim.run()

        E2[nb, quad][k] = \
            norm((olim.U - u_).flatten(), np.inf)/norm(u_.flatten(), np.inf)

################################################################################
# 3D

s3d = lambda x, y, z: 1/(2 + vx*x + vy*y + vz*z)
s_1, s_2 = s3d(x_fac_1, y_fac_1, z_fac_1), s3d(x_fac_2, y_fac_2, z_fac_2)

def make_u3d(x_fac, y_fac, z_fac, vx, vy, vz):
    return lambda x, y, z: \
        (1/np.sqrt(vx**2 + vy**2 + vz**2)) * \
        np.arccosh(
            1 +
            s3d(x_fac, y_fac, z_fac)*s3d(x, y, z)*(vx**2 + vy**2 + vz**2)*
            ((x - x_fac)**2 + (y - y_fac)**2 + (z - z_fac)**2)/2)

u3d_1 = make_u3d(x_fac_1, y_fac_1, z_fac_1, vx, vy, vz)
u3d_2 = make_u3d(x_fac_2, y_fac_2, z_fac_2, vx, vy, vz)

u3d = lambda x, y, z: np.minimum(u3d_1(x, y, z), u3d_2(x, y, z))

E3 = dict()

for nb, quad in Olims3:
    print('%s %s' % (nb, quad))

    E3[nb, quad] = np.zeros(len(N3D))

    for a, n in enumerate(N3D):
        print('- n = %d (%d/%d)' % (n, a + 1, len(N3D)))

        L = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(L, L, L)
        u_ = u3d(X, Y, Z)
        S = s3d(X, Y, Z)

        h = 1/(n - 1)
        i_1, j_1, k_1 = y_fac_1/h, x_fac_1/h, z_fac_1/h
        i_2, j_2, k_2 = y_fac_2/h, x_fac_2/h, z_fac_2/h

        olim = pyolim.Olim(nb, quad, S, h)

        R_1 = np.sqrt((x_fac_1 - X)**2 + (y_fac_1 - Y)**2 + (z_fac_1 - Z)**2)
        fs_1 = pyolim.FacSrc((i_1, j_1, k_1), s_1)
        for i, j, k in zip(*np.where(R_1 <= R_fac)):
            olim.set_fac_src((i, j, k), fs_1)
        olim.add_src((x_fac_1, y_fac_1, z_fac_1), s_1)

        R_2 = np.sqrt((x_fac_2 - X)**2 + (y_fac_2 - Y)**2 + (z_fac_2 - Z)**2)
        fs_2 = pyolim.FacSrc((i_2, j_2, k_2), s_2)
        for i, j, k in zip(*np.where(R_2 <= R_fac)):
            olim.set_fac_src((i, j, k), fs_2)
        olim.add_src((x_fac_2, y_fac_2, z_fac_2), s_2)

        olim.run()

        E3[nb, quad][a] = \
            norm((u_ - olim.U).flatten(), np.inf)/norm(u_.flatten(), np.inf)

################################################################################
# Plotting (numerical results figure)

fig, axes = plt.subplots(1, 2, sharex='col', sharey='all', figsize=(6.5, 2.5))

axes[0].set_ylabel(r'$\|u - U\|_\infty/\|u\|_\infty$')

ax = axes[0]
for Olim in Olims2:
    ax.loglog(N, E2[Olim], label=common2.get_marcher_plot_name(*Olim),
              linewidth=1, marker='|', markersize=3.5)
ax.minorticks_off()
N_pow_2d = np.arange(args.min_2d_power, args.max_2d_power + 1, 3)
ax.set_xticks(2**N_pow_2d + 1)
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in N_pow_2d])
ax.set_xlabel('$N$')

ax.legend(loc='lower left', prop={'size': 8})

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cmap = [0, 1, 4, 3]
linestyles = ['-', '--', ':']

ax = axes[1]
it = 0
for Olim in Olims3:
    ax.loglog(N3D, E3[Olim], label=common3.get_marcher_plot_name(*Olim),
              color=colors[cmap[it//3]], linestyle=linestyles[it % 3],
              linewidth=1, marker='|', markersize=3.5)
    it += 1
ax.minorticks_off()
N_pow_3d = np.arange(args.min_3d_power, args.max_3d_power + 1, 3)
ax.set_xticks(2**N_pow_3d + 1)
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in N_pow_3d])
ax.set_xlabel('$N$')

ax.legend(loc='lower left', ncol=1, prop={'size': 8})

fig.tight_layout()
fig.show()

fig.savefig('qv_plots.eps')
