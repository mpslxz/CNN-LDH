import features
import filterBuilder
import numpy as np
from PIL import ImageOps, Image

def extractFeatures(img):
    img = ImageOps.fit(Image.fromarray(img), (32, 32))
    hMats, wFilters = filterBuilder.buildFilters()
    return features.features(np.asarray(img),hMats,wFilters)