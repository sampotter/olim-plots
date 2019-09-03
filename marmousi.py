import numpy as np

from skimage.transform import rescale

import pyolim


xmin, xmax = 0.0, 9.2
zmin, zmax = 0.0, 3.0


def load_velocity():

    # Load Marmousi velocity model (initially in [m/s])
    # The model is 9.2km long and 3km deep
    with np.load('marmousi.npz') as data:
        F = data['F']

    # Convert to [km/s]
    F /= 1000
    
    return F


def load_scaled_velocity(lmin=-3, lmax=3):
    
    Fs = [load_velocity()]
    
    kwargs = dict(
        mode='constant',
        multichannel=False,
        anti_aliasing=False
    )
    
    for l in range(lmin, 0):
        Fs.append(rescale(Fs[-1], 1/2, **kwargs))
    
    Fs = list(reversed(Fs))
    
    for l in range(lmax):
        Fs.append(rescale(Fs[-1], 2, **kwargs))

    return Fs


def load_scaled_slowness(lmin=-3, lmax=3):
    
    return [1/F for F in load_scaled_velocity(lmin, lmax)]


def solve_point_source_problem(S, nb, quad):
    
    nz = S.shape[0]
    h = zmax/(nz - 1)

    olim = pyolim.Olim(nb, quad, S, h)
    olim.add_src((0,) * S.ndim)
    olim.solve()
    
    return olim