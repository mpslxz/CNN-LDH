import numpy as np
import partition
import wedgeVariance
from scipy.misc import imresize

def features(img, hMats, wFilters):
    imageSize = 16
    maxScale = 3
    out = []

    img = imresize(img, (32, 32))
    for i in range(1,maxScale+1):
        had = partition.partitionHadImg(img, i, hMats[i-1])
        for j in range(0,np.asarray(had).shape[0]):
            v1, v2, v3 = wedgeVariance.wedgeVariance(had[j], wFilters[3*(i-1)], wFilters[3*(i-1) + 1], wFilters[3*(i-1) + 2])
            out.append([v1,v2,v3])

    return out