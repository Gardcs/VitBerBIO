import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from simulator import plane, normalize, photon, unitSphericalDistribution, crossProd

#pl = plane(np.array((0.7, 0.0, 0.0)), normalize(np.array((-1.0, -1.0, 0.0))))
#photon.planes.append(pl)





fig = plt.figure(0)
ax = plt.axes(xlim=(-3,3), ylim=(-3,3))
dots, = ax.plot([], [], 'kx', markersize=1)

def init():
    dots.set_data([], [])
    return dots

def animate(i):
    xrand = (np.random.rand(10)*2-1)*np.sin(i/10)
    yrand = np.random.rand(10)*2-1
    dots.set_data(xrand, yrand)
    return dots

frames = 240
def expose(i):
    theta = i *2*np.pi/frames
    r = np.array([np.cos(theta), np.sin(theta), 0])
    normal = r
    basis = [crossProd(r, np.array([0,0,1])), np.array([0,0,1]), normal]
    pl = plane(r, normalize(normal), basis=basis)
    photon.planes = [pl];
    print("!", end="")
    for i in range(500000):
        dir = unitSphericalDistribution()
        ph = photon(-r, dir)
        ph.jitPrimer()
    dots.set_data(*np.array(pl.markings).T)
    del pl
    print("*", end="")
    return dots

anim = animation.FuncAnimation(fig, expose, interval=20, frames=frames, init_func=init, blit=False)
anim.save("anim.gif", fps=30);
#plt.show()


