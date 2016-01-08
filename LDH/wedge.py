import numpy as np
import math
def wedge(patch, percent, dir):

    M = patch.shape[0]
    x = math.ceil((percent / 50.0) * M)
    W = np.zeros((M,M))

    for i in range(0,M):
        for j in range(0,M):
            if (j <= i*(x/M)):
                if(dir == 1):
                    W[i,j] = 1
                else:
                    W[j,i] = 1



    W[0,0] = 0
    return W
