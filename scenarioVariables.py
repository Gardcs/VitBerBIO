xLim = (-3.0, 3.0);
yLim = (-3.0, 3.0);
zLim = (-3.0, 3.0);
timestep = 1e-10;
c = 3e8

K1 = 6e9
K2 = 0.02e9

from numba import jit
import math

@jit(nopython=True)
def dissipation_density(y,x,z):
    return K1*timestep if ((math.sqrt(x**2+z**2)-0.3)**2+y**2<0.0025 and z < 0) or (math.sqrt((abs(x)-0.15)**2+y**2+(z-0.2)**2) < 0.07) else 0;