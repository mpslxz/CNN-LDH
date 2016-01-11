
import cPickle
import gzip
import loadDataset
import loadSegmentationDataset
import gc
import pickle

import numpy as np
import os
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
import makeDataOnTheFly

def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


os.system('clear')
GPU = True
if GPU:
    print "Trying to run under a GPU."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU."

print "\nRunning on " + theano.config.device + "...\n"


def load_data_shared(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        self.LDH = T.matrix("LDH")
        self.LDH = self.x[:, 1764:2016]
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x[:,0:1764], self.x[:,0:1764], self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            if (j == 2):
                # inpt = T.concatenate((prev_layer.output, self.LDH))
                layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size, self.LDH)
            else:
                layer.set_inpt(
                    prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):

        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # training_x = training_x_LDH[:,0:1764]
        # validation_x  = validation_x_LDH[:,0:1764]
        # test_x = test_x_LDH[:,0:1764]

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}\n".format(iteration))
                cost_ij = train_mb(minibatch_index)
                # print cost_ij
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy > best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        with open("log_balanced.txt","a") as logFile:
            logFile.write("Finished training network.")
            logFile.write("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
            logFile.write("Corresponding test accuracy of {0:.2%}\n".format(test_accuracy))

    def predict(self, testData):
        counter = T.lscalar()
        predFcn = theano.function([counter], self.layers[-1].y_out, givens={self.x: testData[counter*self.mini_batch_size:(counter+1)*self.mini_batch_size]})
        result = np.zeros((64*64))

        for i in range(0,(64*64/self.mini_batch_size)):
            result[i*self.mini_batch_size:(i+1)*self.mini_batch_size] = predFcn(i)
        return result

#### Define layer types

class ConvPoolLayer(object):


    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size, LDH):
        LDHFeatures = LDH.reshape((mini_batch_size, 252))
        CFeatures = inpt.reshape((mini_batch_size, 810))
        self.inpt = T.concatenate([CFeatures, LDHFeatures])#inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))



def size(data):
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

def sharedForPrediction(data):
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x

mini_batch_size = 10
training_data, validation_data, test_data = loadSegmentationDataset.loadData(dim=64)

print "Dataset loaded successfully...\n"

print "Building the model...\n"

# net = Network([
#         ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                       filter_shape=(20, 1, 5, 5),
#                       poolsize=(2, 2), activation_fn=ReLU),
#         ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                       filter_shape=(40, 20, 5, 5),
#                       poolsize=(2, 2), activation_fn=ReLU),
#         FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#         SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)


net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 42, 42),
                      filter_shape=(10, 1, 3, 3),
                      poolsize=(2, 2), activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 10, 20, 20),
                      filter_shape=(10, 10, 3, 3),
                      poolsize=(2, 2), activation_fn=ReLU),
        # ConvPoolLayer(image_shape=(mini_batch_size, 5, 29, 29),
        #               filter_shape=(5, 5, 3, 3),
        #               poolsize=(3, 3), activation_fn=ReLU),
        FullyConnectedLayer(n_in=10*9*9 + 252, n_out=2000, activation_fn=ReLU),
        SoftmaxLayer(n_in=2000, n_out=2)], mini_batch_size)


# prediction = test_data

# l[:,0].tofile("testLabel.txt"," ")
print "Training the model...\n"
net.SGD(training_data, 200, mini_batch_size, 0.02, validation_data, test_data, 0.02)

# for rep in range(1,9):
#    training_data = None
#    validation_data = None
#    test_data = None
#    gc.collect()
#    training_data, validation_data, test_data = loadSegmentationDataset.loadData(dim=64)
#    net.SGD(training_data, 200, mini_batch_size, 0.02, validation_data, test_data, 0.02)
#
# f = file("trained_net_trainedOnJordensImagesSmallLambda.p","w")
# pickle.dump(net,f)
#
# # f = file("trained_net.p",'rb')
# # Net = pickle.load(f)
#
# print "Training completed...\n"
#
# print "Predicting...\n"
#
# for i in range(1, 1442):
#     print i
#     imPath = "/home/mehran/Desktop/left/Im%d.jpg" %i
#     imEnhanPath = "/home/mehran/Desktop/left/normEnhanIm%d.jpg" %i
#     sample = makeDataOnTheFly.makeDataOnTheFly(64, imPath, imEnhanPath)
#     predData = sharedForPrediction(sample)
#     predRes = net.predict(predData)
#     saveFile = open("/home/mehran/Desktop/left/Predictions_JordensImagesSmallLambda/Prediction%d.txt" %i, "a")
#     np.savetxt(saveFile,predRes,'%i'," ")
#     saveFile.close()
#
# for i in range(1, 1435):
#     print i
#     imPath = "/home/mehran/Desktop/right1/Im%d.jpg" %i
#     imEnhanPath = "/home/mehran/Desktop/right1/normEnhanIm%d.jpg" %i
#     sample = makeDataOnTheFly.makeDataOnTheFly(64, imPath, imEnhanPath)
#     predData = sharedForPrediction(sample)
#     predRes = net.predict(predData)
#     saveFile = open("/home/mehran/Desktop/right1/Predictions_JordensImagesSmallLambda/Prediction%d.txt" %i, "a")
#     np.savetxt(saveFile,predRes,'%i'," ")
#     saveFile.close()
#
# for i in range(1, 1598):
#     print i
#     imPath = "/home/mehran/Desktop/right2/Im%d.jpg" %i
#     imEnhanPath = "/home/mehran/Desktop/right2/normEnhanIm%d.jpg" %i
#     sample = makeDataOnTheFly.makeDataOnTheFly(64, imPath, imEnhanPath)
#     predData = sharedForPrediction(sample)
#     predRes = net.predict(predData)
#     saveFile = open("/home/mehran/Desktop/right2/Predictions_JordensImagesSmallLambda/Prediction%d.txt" %i, "a")
#     np.savetxt(saveFile,predRes,'%i'," ")
#     saveFile.close()
#
