xLim = (-1.0, 1.0);
yLim = (-1.0, 1.0);
zLim = (-1.0, 1.0);
timestep = 1e-11;
c = 3e8

K1 = 0.2e9
K2 = 0.02e9

from numba import jit

@jit(nopython=True)
def dissipation_density(x,y,z):
    return K1*timestep if z>0 else K2*timestep