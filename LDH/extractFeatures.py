import features
import filterBuilder

def extractFeatures(img):
    hMats, wFilters = filterBuilder.buildFilters()
    return features.features(img,hMats,wFilters)