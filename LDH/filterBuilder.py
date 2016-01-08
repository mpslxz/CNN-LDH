import numpy as np
import hadamard
import wedge

def buildFilters():
    hMats = []
    hMats.append(hadamard.walsh(16))
    hMats.append(hadamard.walsh(8))
    hMats.append(hadamard.walsh(4))

    wFilters = []
    wFilters.append(wedge.wedge(np.zeros((16,16)), 10, 1))
    wFilters.append(wedge.wedge(np.zeros((16,16)), 10, 2))
    W = 1 - wFilters[-1] - wFilters[-2]
    W[0,0] = 0
    wFilters.append(W)

    wFilters.append(wedge.wedge(np.zeros((8,8)), 10, 1))
    wFilters.append(wedge.wedge(np.zeros((8,8)), 10, 2))
    W = 1 - wFilters[-1] - wFilters[-2]
    W[0,0] = 0
    wFilters.append(W)

    wFilters.append(wedge.wedge(np.zeros((4,4)), 10, 1))
    wFilters.append(wedge.wedge(np.zeros((4,4)), 10, 2))
    W = 1 - wFilters[-1] - wFilters[-2]
    W[0,0] = 0
    wFilters.append(W)

    return (hMats, wFilters)
