import numpy as np

def wedgeVariance(hadam, W1, W2, W3):
    H = np.asarray(hadam + 0.5)
    buf = W1 * H
    rInd, cInd = np.nonzero(buf)
    V1 = []
    for i in range(0,rInd.shape[0]): V1.append(buf[rInd[i],cInd[i]])

    buf = W2 * H
    rInd, cInd = np.nonzero(buf)
    V2 = []
    for i in range(0,rInd.shape[0]): V2.append(buf[rInd[i],cInd[i]])

    buf = W3 * H
    rInd, cInd = np.nonzero(buf)
    V3 = []
    for i in range(0,rInd.shape[0]): V3.append(buf[rInd[i],cInd[i]])

    return (np.var(np.asarray(V1)), np.var(np.asarray(V2)), np.var(np.asarray(V3)))
