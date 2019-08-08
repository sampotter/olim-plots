BUILD_TYPE='Release'

import pyolim
import numpy as np
import time

from pyolim import Neighborhood, Quadrature

# marchers = [pyolim.BasicMarcher,
#             pyolim.Olim4Mid0, pyolim.Olim4Mid1, pyolim.Olim4Rect,
#             pyolim.Olim8Mid0, pyolim.Olim8Mid1, pyolim.Olim8Rect]

# olim4_marchers = [pyolim.Olim4Mid0, pyolim.Olim4Mid1, pyolim.Olim4Rect]
# olim8_marchers = [pyolim.Olim8Mid0, pyolim.Olim8Mid1, pyolim.Olim8Rect]

# mid0_marchers = [pyolim.Olim4Mid0, pyolim.Olim8Mid0]
# mid1_marchers = [pyolim.Olim4Mid1, pyolim.Olim8Mid1]
# rect_marchers = [pyolim.Olim4Rect, pyolim.Olim8Rect]

# _marcher_names = {
#     pyolim.BasicMarcher: 'basic',
#     pyolim.Olim4Mid0: 'olim4 mp0',
#     pyolim.Olim4Mid1: 'olim4 mp1',
#     pyolim.Olim4Rect: 'olim4 rhr',
#     pyolim.Olim8Mid0: 'olim8 mp0',
#     pyolim.Olim8Mid1: 'olim8 mp1',
#     pyolim.Olim8Rect: 'olim8 rhr'}

# def get_marcher_name(marcher):
#     return _marcher_names[marcher]

# _marcher_plot_names = {
#     pyolim.BasicMarcher: '\\texttt{basic}',
#     pyolim.Olim4Mid0: '\\texttt{olim4\_mp0}',
#     pyolim.Olim4Mid1: '\\texttt{olim4\_mp1}',
#     pyolim.Olim4Rect: '\\texttt{olim4\_rhr}',
#     pyolim.Olim8Mid0: '\\texttt{olim8\_mp0}',
#     pyolim.Olim8Mid1: '\\texttt{olim8\_mp1}',
#     pyolim.Olim8Rect: '\\texttt{olim8\_rhr}'}

# def get_marcher_plot_name(marcher):
#     return _marcher_plot_names[marcher]

_neighborhood_plot_names = {
    Neighborhood.OLIM4: 'olim4',
    Neighborhood.OLIM8: 'olim8',
}

_quadrature_plot_names = {
    Quadrature.MP0: 'mp0',
    Quadrature.MP1: 'mp1',
    Quadrature.RHR: 'rhr',
}

def get_marcher_plot_name(nb, quad=Quadrature.RHR):
    if nb == Neighborhood.FMM2:
        s = r'fmm2d'
    else:
        s = _neighborhood_plot_names[nb] + r'\_' + _quadrature_plot_names[quad]
    return r'\texttt{' + s + r'}'

# _marchers_by_name = {v: k for k, v in _marcher_names.items()}

# def get_marcher_by_name(name):
#     return _marchers_by_name[name]

def relerr(x, y, ord_):
    norm = lambda x: np.linalg.norm(x.flat, ord_)
    distxy = norm(x - y)
    return max(distxy/norm(x), distxy/norm(y))

def get_exact_soln(f, M):
    l = np.linspace(-1, 1, M)
    return f(*np.meshgrid(l, l))

def compute_soln(marcher, s, M):
    l = np.linspace(-1, 1, M)
    m = marcher(s(*np.meshgrid(l, l)), 2/(M - 1))
    m.addBoundaryNode(int(M/2), int(M/2))
    m.run()
    U = np.array([[m.getValue(i, j) for j in range(M)] for i in range(M)])
    return U

def tic():
    tic.t0 = time.time()
tic.t0 = None

def toc():
    if tic.t0:
        return time.time() - tic.t0
    else:
        raise RuntimeError("tic() hasn't been called")

def time_marcher(Marcher, s, n, ntrials=10):
    print('  - n = %d' % n)
    h = 2/(n - 1)
    l = np.linspace(-1, 1, n)
    i = int(n/2)
    x, y = np.meshgrid(l, l)
    s_cache = s(x, y)
    def do_trial(trial):
        tic()
        m = Marcher(s_cache, h)
        m.addBoundaryNode(i, i)
        m.run()
        return toc()
    times = min(do_trial(t) for t in range(ntrials))
    return times
