import math
import numpy as np

def partition_img(img, scale, transformMat):
    out = []
    transformed = []

    M = int(img.shape[0]) - 1
    step = int((M+1) / (math.pow(2, scale)))
    for i in range(0,M,step):
        for j in range(0,M,step):
            out.append(img[i:i+step, j:j+step])
            transformed.append(np.mat(transformMat) * np.mat(out[-1]) * np.mat(transformMat))

    return transformed
