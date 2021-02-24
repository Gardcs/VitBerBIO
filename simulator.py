# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:04:37 2021

@author: astap
"""
timestep = 1e-11;
c = 3e8


import numpy as np

import jitSpeedup as speedUp
from numba import jit
from scenarioVariables import xLim, yLim, zLim, c, timestep

isInLim = lambda D1Limits, D1Position: D1Limits[0]<D1Position<D1Limits[1]
enclosed = lambda D3Position: isInLim(xLim, D3Position[0]) & isInLim(yLim, D3Position[1]) & isInLim(zLim, D3Position[2])
crossProd = lambda a, b: np.array(((a[1]*b[2]-a[2]*b[1]), (a[2]*b[0]-a[0]*b[2]), (a[0]*b[1]-a[1]*b[0])))
dotProd = lambda a,b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
norm = lambda a: np.sqrt(dotProd(a,a))
orthoProjection = lambda a, b: a - b*dotProd(a,b)/(norm(b)**2)
normalize = lambda a: a/norm(a);

def generateUnitBasis(normal):
    zetaHat = normalize(normal)
    iHat = np.array([1,0,0]); jHat = np.array([0,1,0])
    if(dotProd(zetaHat, iHat) > 1/np.sqrt(2)):
        xiHat = orthoProjection(iHat, zetaHat)
    else:
        xiHat = orthoProjection(jHat, zetaHat)
    xiHat /= norm(xiHat)
    etaHat = crossProd(zetaHat, xiHat)
    return np.array((zetaHat, xiHat, etaHat))


def unitSphericalDistribution():
    latDist = np.random.random()*2 - 1
    sgn = np.sign(latDist); latDist *= sgn
    latitude = np.pi*(np.sqrt(latDist)/2 if sgn>0 else 1-np.sqrt(latDist)/2)
    longitude = 2*np.pi*np.random.random()
    x,y,z = np.sin(latitude)*np.cos(longitude), np.sin(latitude)*np.sin(longitude), np.cos(latitude)
    return x,y,z

class plane:
    def __init__(self, location, direction):
        self.zeta, self.xi, self.eta = generateUnitBasis(direction)
        self.location = location
        self.markings = []
        #skaper en basis (xi, eta, zeta) for planet med 'direction'(=zeta) som enhetsvektor
        #i retning av planets normal-akse. Planet defineres
        #til å krysse location, og baserer sine koordinater ut fra dette


    def __le__(self, photon):
        relativeLocationInitial = photon.location - self.location
        zetaCoordinateInitial = dotProd(relativeLocationInitial, self.zeta)
        relativeLocationFinal = photon.nextStep() - self.location
        zetaCoordinateFinal = dotProd(relativeLocationFinal, self.zeta)
        return np.sign(zetaCoordinateFinal) != np.sign(zetaCoordinateInitial)
        #Returnerer sann dersom fortegnet til zeta-koordinatet endres
        #etter ett timestep. Nå vil  'plan <= foton' returnere sann
        #dersom fotonet krysser planet i løpet av neste timestep

    def __lt__(self, photon):
        if not (self <= photon): #Sjekker om det skjer skæring mellom plan og foton
            return False #Dersom ingen skjæring: Returner false
        relativeLocationInitial = photon.location - self.location
        planePhotonDistance = dotProd(relativeLocationInitial, self.zeta)
        requiredTravelDistance = planePhotonDistance/dotProd(photon.direction, self.zeta)
        inPlanePosition = photon.nextStep(distance=requiredTravelDistance);
        relativeLocationFinal = inPlanePosition - self.location
        self.markings.append(np.array((dotProd(relativeLocationFinal, self.xi), dotProd(relativeLocationFinal, self.eta))))
        return True



class photon:
    planes = []; #Alle aktuelle plan hvilket fotonet kan krysse

    def __init__(self, location, direction):
        self.location = location
        self.direction = normalize(direction)

    def nextStep(self, distance=c*timestep):
        return self.location + self.direction * distance

    def jitPrimer(self):
        planesCoordinates = [plane.location for plane in photon.planes]
        planesDirections = [plane.zeta for plane in photon.planes]
        initialCoordinates = self.location
        initialDirection = self.direction
        hitPlane, position = speedUp.jitTilHit(planesCoordinates, planesDirections,
                          initialCoordinates, initialDirection)

        if not hitPlane:
            del self
            return position
        self.location = position


        for plane in photon.planes:
            if(plane < self):
                del self
                return position

        raise Exception("Disagreement of hit between jit-nopython and python")




pl = plane(np.array((0.7,0.0,0.0)), normalize(np.array((-1.0,-1.0,0.0))))
photon.planes.append(pl)

poses = []
for j in range(4):
    for i in range(1000):
        dir = unitSphericalDistribution()
        pos = np.array((-.9,-.9,.0))
        ph = photon(pos.copy(), dir)
        current_poses = [pos, ph.jitPrimer()]
        poses.append(np.array(current_poses))
    print("*", end='')

poses = poses[::4]






import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(0)
ax = fig.add_subplot(111,projection='3d')
for path in poses:
    ax.plot(*path.T, ':', c="k")

scatterpos = []
for mrk in pl.markings:
    pos = pl.location + pl.xi*mrk[0] + pl.eta*mrk[1]
    scatterpos.append(pos)
scatterpos = np.array(scatterpos)
ax.scatter(*scatterpos.T,c='r',marker='x')
plt.show()
fig.savefig("illustrasjon.pdf")

fig2 = plt.figure(1)
plt.plot(*np.array(pl.markings).T, 'rx')

plt.show()