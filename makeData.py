import sys
import pickle
import numpy as np
from math import ceil
sys.path.insert(0, 'LDH')
import extractFeatures
from PIL import Image, ImageOps
import timeit


def addPaddingZero(img,dim):
    newDim = ceil(5*dim/3)
    newImage = np.zeros((newDim,newDim))
    newImage[ceil(dim/3):ceil(4*dim/3),ceil(dim/3):ceil(4*dim/3)] = img
    return  newImage #Image.fromarray(newImage)

def makeData(dim):
    imageCount = 580
    size = (dim, dim)
    trainingPercent = 0.7
    testPercent = 0.1
    validationPercent = 0.2
    winDim = ceil(2*dim/3)
    laminae = np.ndarray((dim*dim, winDim*winDim + 84*3))

    # labels = np.zeros((dim*dim, 2))
    labels = np.zeros((dim*dim))

    fileCount = 0

    print "Loading Laminae...\n"
    for i in range(1, imageCount+1):
        print i
        # img = Image.open("newLam/img_%d.jpg" %i)
        img = Image.open("/home/mehran/Desktop/ConvNet/newLam/img_%d.jpg" %i)
        img = ImageOps.fit(img, size)
        # imgEn = Image.open("enhanced/img_%d.jpg" %i)
        imgEn = Image.open("/home/mehran/Desktop/ConvNet/enhanced/img_%d.jpg" %i)
        imgEn = ImageOps.fit(img, size)

        segmented = Image.open("/home/mehran/Desktop/ConvNet/base of lamina/lamBase_%d.jpg" %i)
        segmented = ImageOps.fit(segmented, size)

        paddedImage = addPaddingZero(img, dim)
        paddedImageEn = addPaddingZero(imgEn, dim)
         #paddedSegmented = addPaddingZero(segmented, dim)
        count = 0
        for m in range(0, dim):
            for n in range(0, dim):
                window = paddedImage[m:m+winDim, n:n+winDim]
                windowEn = paddedImageEn[m:m+winDim, n:n+winDim]
                Res = 0.7*windowEn + 0.3*window
                laminae[count, :] = np.append(np.reshape(Res, winDim*winDim)/np.amax(Res), np.asarray(extractFeatures.extractFeatures(Res/np.amax(Res))))
                S = segmented.load()

                labels[count] = 1 if S[m, n] > 0 else 0
                 # labels[count, 0] = 1 if S[m, n] > 0 else 0
                 # labels[count, 1] = 1 - S[m, n]
                count += 1

        # f = file("leftImgs/Pickles/sample_%d.p" %i, 'wb')
        # pickle.dump(laminae, f)
     #
        posIndices = np.nonzero(labels)[0]
        posSampleSize = np.count_nonzero(labels)
        if(posSampleSize > 0):
            print i
            negIndices = np.random.randint(0, dim*dim, posSampleSize)
            a = [labels[v] for v in posIndices]
            b = [labels[u] for u in negIndices]

            L = np.hstack((a, b))
            Patches = np.vstack(([laminae[v,:] for v in posIndices], [laminae[u,:] for u in negIndices]))
            f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %fileCount, 'wb')
            pickle.dump([Patches, L], f)
            fileCount += 1


makeData(64)
