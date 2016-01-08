
import numpy as np
from math import ceil
from PIL import Image, ImageOps


def addPaddingZero(img,dim):
    newDim = ceil(5*dim/3)
    newImage = np.zeros((newDim,newDim))
    newImage[ceil(dim/3):ceil(4*dim/3),ceil(dim/3):ceil(4*dim/3)] = img
    return  newImage

def makeDataOnTheFly(dim, imPath, imEnhanPath):
    size = (dim, dim)
    winDim = ceil(2*dim/3)
    laminae = np.ndarray((dim*dim, winDim*winDim))


    img = Image.open(imPath)
    img = ImageOps.fit(img, size)
    imgEn = Image.open(imEnhanPath)
    imgEn = ImageOps.fit(img, size)

    paddedImage = addPaddingZero(img, dim)
    paddedImageEn = addPaddingZero(imgEn, dim)

    count = 0
    for m in range(0, dim):
        for n in range(0, dim):
            window = paddedImage[m:m + winDim, n:n + winDim]
            windowEn = paddedImageEn[m:m + winDim, n:n + winDim]
            Res = 0.7 * windowEn + 0.3 * window
            laminae[count, :] = np.reshape(Res, winDim * winDim) / np.amax(Res)
            count += 1

    return laminae