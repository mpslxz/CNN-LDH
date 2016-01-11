__author__ = 'mehran'

import pickle
import numpy as np
from math import ceil
import theano
from theano import tensor as T
from PIL import Image, ImageOps

def loadData(dim):
    imageCount = 580
    size = (dim, dim)
    trainingPercent = 0.7
    testPercent = 0.1
    validationPercent = 0.2
    winDim = ceil(2*dim/3)
    laminae = np.ndarray((dim*dim, winDim*winDim))

    labels = np.zeros((dim*dim))

    fileCount = 0
    numOfTrainingSamples = 5
    numOfTestSamples = 2
    numOfValidationSamples = 2

    print "Making training data...\n"
    ind = np.random.randint(0, 573)
    # f = file("/home/mehran/Desktop/left/Pickles/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+verBody_dilated_labels/sample_balanced_%d.p" %ind, 'rb')
    f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_bone/bone_sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/full_enhanced+bones/sample_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+bone_labels/sample_balanced_%d.p" %ind, 'rb')
    [lam, l] = pickle.load(f)
    training_data = lam
    training_labels = l
    for i in range(1, numOfTrainingSamples):
        ind = np.random.randint(0, 573)
        print i
        # f = file("/home/mehran/Desktop/left/Pickles/sample_balanced_%d.p" %ind, 'rb')
        f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+verBody_dilated_labels/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_bone/bone_sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/full_enhanced+bones/sample_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+bone_labels/sample_balanced_%d.p" %ind, 'rb')
        [lam, l] = pickle.load(f)
        training_data = np.vstack((training_data,lam))
        training_labels = np.hstack((training_labels,l))
        f.close()

    print "Making test data...\n"
    ind = np.random.randint(0, 573)
    f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %ind, 'rb')
    # f = file("/home/mehran/Desktop/left/Pickles/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+verBody_dilated_labels/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_bone/bone_sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/full_enhanced+bones/sample_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+bone_labels/sample_balanced_%d.p" %ind, 'rb')
    [lam, l] = pickle.load(f)
    test_data = lam
    test_labels = l
    for i in range(1, numOfTestSamples):
        ind = np.random.randint(0, 573)
        print i
        f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %ind, 'rb')
        # f = file("/home/mehran/Desktop/left/Pickles/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+verBody_dilated_labels/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_bone/bone_sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/full_enhanced+bones/sample_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+bone_labels/sample_balanced_%d.p" %ind, 'rb')
        [lam, l] = pickle.load(f)
        test_data = np.vstack((test_data,lam))
        test_labels = np.hstack((test_labels,l))
        f.close()


    print "Making validation data...\n"
    ind = np.random.randint(0, 573)
    f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %ind, 'rb')
    # f = file("/home/mehran/Desktop/left/Pickles/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+verBody_dilated_labels/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels/sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_bone/bone_sample_balanced_%d.p" %ind, 'rb')
    # f = file("Pickles/full_enhanced+bones/sample_%d.p" %ind, 'rb')
    # f = file("Pickles/balanced/enhanced_overlay+bone_labels/sample_balanced_%d.p" %ind, 'rb')
    [lam, l] = pickle.load(f)
    validation_data = lam
    validation_labels = l
    for i in range(1, numOfValidationSamples):
        ind = np.random.randint(0, 573)
        print i
        f = file("/home/mehran/Desktop/ConvNet/Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels_with_LDH/sample_balanced_%d.p" %ind, 'rb')
        # f = file("/home/mehran/Desktop/left/Pickles/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+verBody_dilated_labels/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+lamBase_dilated_new_labels/sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_bone/bone_sample_balanced_%d.p" %ind, 'rb')
        # f = file("Pickles/full_enhanced+bones/sample_%d.p" %ind, 'rb')
        # f = file("Pickles/balanced/enhanced_overlay+bone_labels/sample_balanced_%d.p" %ind, 'rb')
        [lam, l] = pickle.load(f)
        validation_data = np.vstack((validation_data,lam))
        validation_labels = np.hstack((validation_labels,l))
        f.close()

    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return shared([training_data, training_labels]), shared([test_data, test_labels]), shared([validation_data, validation_labels])


# training_data, test_data, validation_data =loadData(dim=64)

# print "done!"
# img = Image.open("lamina/img_1.jpg")
# img = ImageOps.fit(img, (128,128))
# newImg = img.load()
# print newImg[0,0]
# print L[573]
# img = D[573, :, :]/256
# pylab.gray()
# pylab.imshow(newImg)
# pylab.show()



