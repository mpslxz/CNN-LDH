__author__ = 'mehran'

import numpy as np
import theano
from theano import tensor as T
from PIL import Image, ImageOps


def loadData(dim):
    size = (dim, dim)
    trainingPercent = 0.7
    testPercent = 0.1
    validationPercent = 0.2
    laminae = np.ndarray((1149, dim*dim))
    notLaminae = np.ndarray((1149, dim*dim))

    print "Loading Laminae...\n"
    for i in range(1,1150):
        img = Image.open("lamina/img_%d.jpg" %i)
        img = ImageOps.fit(img,dim)
        # img.thumbnail(size, Image.ANTIALIAS)
        laminae [i-1, :] = np.reshape(np.asarray(img), dim*dim)/np.amax(np.asarray(img))

    print "Loading Not-laminae...\n"
    for i in range(1,1150):
        img = Image.open("notLamina/IMG_%d.jpg" %i)
        img = ImageOps.fit(img,size)
        # img.thumbnail(size, Image.ANTIALIAS)
        notLaminae[i-1, :] = np.reshape(np.asarray(img), dim*dim)/np.amax(np.asarray(img))

    print "Stacking data...\n"
    data = np.vstack((laminae, notLaminae))
    labels = np.hstack((np.ones(1149), np.zeros(1149)))
    # l1 = np.hstack((np.ones(574), np.zeros(574)))
    # l2 = np.hstack((np.zeros(574), np.ones(574)))
    # labels = np.vstack([l1, l2])

    print "Making training data...\n"
    ind = np.random.randint(0, 2298, 2299 * trainingPercent)
    training_data = data[ind, :]
    training_labels = labels[ind]

    print "Making test data...\n"
    ind = np.random.randint(0, 2298, 2299 * testPercent)
    test_data = data[ind, :]
    test_labels = labels[ind]

    print "Making validation data...\n"
    ind = np.random.randint(0, 2298, 2299 * validationPercent)
    validation_data = data[ind, :]
    validation_labels = labels[ind]

    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return shared([training_data, training_labels]), shared([test_data, test_labels]), shared([validation_data, validation_labels])


# print L[573]
# img = D[573, :, :]/256
# pylab.gray()
# pylab.imshow(img)
# pylab.show()



