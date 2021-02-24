
from numba import jit
import random

from scenario_variables import c, xLim, yLim, zLim, timestep, dissipation_density



@jit(nopython=True)
def jitTilHit(planesCoordinates, planesDirection, initialCoordinates, initialDirection):
    global c, xLim, yLim, zLim, timestep, dissipation_density

    def dotProd(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def D3Difference(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def enclosed(position):
        isEnclosed = xLim[0] <= position[0] <= xLim[1]
        isEnclosed &= yLim[0] <= position[1] <= yLim[1]
        isEnclosed &= zLim[0] <= position[2] <= zLim[1]
        return isEnclosed

    def planePhotonCollision(planePosition, photonPosition, planeDirection, photonDirection):
        position = photonPosition.copy()
        initialRelativePos = D3Difference(position, planePosition)
        position[0] += photonDirection[0] * c * timestep
        position[1] += photonDirection[1] * c * timestep
        position[2] += photonDirection[2] * c * timestep
        finalRelativePos = D3Difference(position, planePosition)

        initialZProjection = dotProd(initialRelativePos, planeDirection)
        finalZProjection = dotProd(finalRelativePos, planeDirection)
        return initialZProjection * finalZProjection <= 0


    position = initialCoordinates;
    direction = initialDirection;
    while(True):
        if not enclosed(position):
            return (False, position)

        dissipation_value = dissipation_density(position[0], position[1], position[2])
        if random.random() < dissipation_value:
            return (False, position)

        for planeCoordinates, planeDirection in zip(planesCoordinates, planesDirection):
            if(planePhotonCollision(planeCoordinates, position,
                                    planeDirection, direction)):
                #undo ett step fordi jit handler med referanser
                return (True, position)

        position[0] += direction[0] * c * timestep
        position[1] += direction[1] * c * timestep
        position[2] += direction[2] * c * timestep

