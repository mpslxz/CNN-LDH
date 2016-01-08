import copy
from scipy.linalg import hadamard

def walsh(N):
    H = hadamard(N)
    B = copy.copy(H)
    ind = []
    for x in range(N): ind.append(int(bin(N+x^x/2)[:2:-1],2))
    for x in range(0,N): B[x,:] = H[ind[x],:]
    return B
