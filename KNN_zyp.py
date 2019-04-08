
import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='bytes')
    X = datadict[b'data']
    Y = datadict[b'labels']#python2 b

    # X = np.array([hog(rgb2gray(reshapeData(img))) for img in datadict[b'data']])#hog
    X = np.array([hog(rgb2gray(reshapeData(img))) for img in datadict[b'data']])#hog

    # X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")#rgb
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  print("hog\n")
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)  
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

  return Xtr, Ytr, Xte, Yte

def reshapeData(data):
    img = np.zeros((32, 32, 3), 'uint8')
    img[..., 0] = np.reshape(data[:1024], (32, 32))
    img[..., 1] = np.reshape(data[1024:2048], (32, 32))
    img[..., 2] = np.reshape(data[2048:3072], (32, 32))
    return img

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

# Xtr, Ytr, Xte, Yte = load_CIFAR10('./cifar-10-batches-py/') # a magic function we provide
# # flatten out all images to be one-dimensional
# Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
# Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
###
# Subsample the data for more efficient code execution in this exercise
x_train, y_train, x_test, y_test = load_CIFAR10('./cifar-10-batches-py/')
num_training = 20000
mask = range(num_training)
x_train = x_train[mask]
y_train = y_train[mask]

num_test = 5000
mask = range(num_test)

x_test = x_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

# Xtr_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
# Xte_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
print (x_train.shape, x_test.shape)

###
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(x_train, y_train) # train the classifier on the training images and labels
Yte_predict = nn.predict(x_test) # predict labels on the test images
#and now print the classification accuracy, which is the average number
#of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f'%(np.mean(Yte_predict == y_test)))
