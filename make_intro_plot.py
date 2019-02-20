#!/usr/bin/env python3

################################################################################
# parse arguments first

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min_3d_power', type=int, default=3)
parser.add_argument('--max_3d_power', type=int, default=10)
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

R_fac = 0.1
P = np.arange(args.min_3d_power, args.max_3d_power + 1)
N3D = 2**P + 1
vx, vy, vz = 5, 13, 20

x_fac_1, y_fac_1, z_fac_1 = 0.0, 0.0, 0.0
x_fac_2, y_fac_2, z_fac_2 = 0.8, 0.0, 0.0

marchers = [eik.Olim6Rect, eik.Olim3dHuMid0]

################################################################################
# collect timings and errors

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
T3 = dict()

for Olim in marchers:
    print(common3d.get_marcher_name(Olim))

    E3[Olim] = np.empty(len(N3D))
    T3[Olim] = np.empty(len(N3D))

    for a, n in enumerate(N3D):
        print('- n = %d (%d/%d)' % (n, a + 1, len(N3D)))

        L = np.linspace(0, 1, n)
        X, Y, Z = np.meshgrid(L, L, L)
        u_ = u3d(X, Y, Z)
        S = s3d(X, Y, Z)

        h = 1/(n - 1)
        i_1, j_1, k_1 = y_fac_1/h, x_fac_1/h, z_fac_1/h
        i_2, j_2, k_2 = y_fac_2/h, x_fac_2/h, z_fac_2/h

        t = np.inf

        for _ in range(1 if n > 100 else 5):

            m_fac = Olim(S, h)

            R_1 = np.sqrt((x_fac_1 - X)**2 + (y_fac_1 - Y)**2 + (z_fac_1 - Z)**2)
            fc_1 = eik.FacCenter3d(i_1, j_1, k_1, s_1)
            for i, j, k in zip(*np.where(R_1 <= R_fac)):
                m_fac.set_node_fac_center(i, j, k, fc_1)
            m_fac.add_boundary_node(x_fac_1, y_fac_1, z_fac_1, s_1)

            R_2 = np.sqrt((x_fac_2 - X)**2 + (y_fac_2 - Y)**2 + (z_fac_2 - Z)**2)
            fc_2 = eik.FacCenter3d(i_2, j_2, k_2, s_2)
            for i, j, k in zip(*np.where(R_2 <= R_fac)):
                m_fac.set_node_fac_center(i, j, k, fc_2)
            m_fac.add_boundary_node(x_fac_2, y_fac_2, z_fac_2, s_2)

            t0 = time.perf_counter()

            m_fac.run()

            t = min(t, time.perf_counter() - t0)

        print('    + %s' % datetime.timedelta(seconds=t))

        T3[Olim][a] = t

        U_fac = np.array([[[m_fac.get_value(i, j, k) for k in range(n)]
                           for j in range(n)]
                          for i in range(n)])

        E3[Olim][a] = \
            norm((u_ - U_fac).flatten(), np.inf)/norm(u_.flatten(), np.inf)


################################################################################
# Plotting (introduction figure)

style = {
    'linestyle': 'solid',
    'linewidth': 1,
    'marker': '|',
    'markersize': 3.5
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.25))

ax = axes[0]
ax.set_xlabel(r'Time (s.)')
ax.set_ylabel(r'$\|u - U\|_\infty/\|u\|_\infty$')
for i, Olim in enumerate([eik.Olim6Rect, eik.Olim3dHuMid0]):
    ax.loglog(T3[Olim], E3[Olim], label=common3d.get_marcher_plot_name(Olim),
              color=colors[i], **style)
ax.legend(prop={'size': 7})
ax.minorticks_off()

ax = axes[1]
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$t_{\texttt{olim3d\_mp0}}/t_{\texttt{olim6\_rhr}}$')
ax.semilogx(N3D, T3[eik.Olim3dHuMid0]/T3[eik.Olim6Rect], **style)
ax.minorticks_off()
ax.set_xticks(2**P[::2] + 1)
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in P[::2]])

ax = axes[2]
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$E_{\texttt{olim6\_rhr}}/E_{\texttt{olim3d\_mp0}}$')
ax.semilogx(N3D, E3[eik.Olim6Rect]/E3[eik.Olim3dHuMid0], **style)
ax.minorticks_off()
ax.set_xticks(2**P[::2] + 1)
ax.set_xticklabels(['$2^{%d} + 1$' % p for p in P[::2]])

fig.tight_layout()
fig.savefig('intro.eps')
