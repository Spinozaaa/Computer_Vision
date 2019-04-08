
import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.feature import local_binary_pattern

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='bytes')
    x = datadict[b'data']
    Y = datadict[b'labels']#python2 b

    # x = np.array([hog(rgb2gray(reshapeData(img)), orientations=8, pixels_per_cell=(8, 8),
    #                 cells_per_block=(3, 3), block_norm='L2') for img in datadict[b'data']])#hog
    # print("xshape", x.shape)
    # settings for LBP
    # radius = 3
    # n_points = 8 * radius
    # x = np.array([local_binary_pattern(rgb2gray(reshapeData(img)), n_points, radius) for img in datadict[b'data']])#lbp
    x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")#rgb
    Y = np.array(Y)
    return x, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    x, Y = load_CIFAR_batch(f)
    xs.append(x)
    ys.append(Y)  
  xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del x, Y
  xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

  return xtr, Ytr, xte, Yte

def reshapeData(data):
    img = np.zeros((32, 32, 3), 'uint8')
    img[..., 0] = np.reshape(data[:1024], (32, 32))
    img[..., 1] = np.reshape(data[1024:2048], (32, 32))
    img[..., 2] = np.reshape(data[2048:3072], (32, 32))
    return img

def cutData(num_training, num_test):
    x_train, y_train, x_test, y_test = load_CIFAR10('./cifar-10-batches-py/')
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    return x_train, y_train, x_test, y_test

def get_CIFAR10_data(num_training=10000, num_validation=1000, num_test=2000):
    x_train, y_train, x_test, y_test = cutData(num_training + num_validation, num_test)
    # 标准化数据：先求平均图像，再将每个图像都减去其平均图像，这样的预处理会加速后期最优化过程中权重参数的收敛性
    mean_image = np.mean(x_train, axis = 0)
    x_train -= mean_image
    x_test -= mean_image

    mask = list(range(num_training, num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]


    return x_train, y_train, x_val, y_val, x_test, y_test


def softmax(x, y, W, b, reg):
    # print("x",x.shape)
    # print("W",W.shape)
    # print("b",b.shape)
    num_train = x.shape[0]
    b = b.repeat(num_train, axis=0)
    f = x.dot(W) + b

    f -= np.max(f, axis=1, keepdims=True) 
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f

    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    dW = x.T.dot(p - ind)
    db = np.sum(p - ind, axis=0, keepdims=True)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W

    return loss, dW, db 

# 调用该函数以获取我们需要的数据，然后查看数据集大小
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

def get_result(x_train, y_train, x_test, y_test):
    W = np.random.randn(x_train.shape[1], 10) * 0.0001 #rgb
    b = np.random.randn(1, 10) * 0.0001
    # W = np.random.randn(x_train.shape[1], 10) * 0.01 #hog
    # b = np.random.randn(1, 10) * 0.01

    # print('w size', W.size)

    NUM_EPOCHS = 50
    dev_num = int(x_train.shape[0]/50)
    step_size = 0.0000042#rgb
    # step_size = 0.07#hog
    best_w = W
    best_b = b
    lastAcc = 1
    
    reg = 1e-6#rgb
    # reg = 0#hog

    for epoch in range(NUM_EPOCHS):
      for i in range(int(x_train.shape[0]/dev_num)):
        mask = np.random.choice(x_train.shape[0], dev_num, replace=False)
        x_dev = x_train[mask]
        y_dev = y_train[mask]
        loss, dw, db = softmax(x_dev, y_dev, W, b, reg)
        # print('dW: ', dw)
        W = np.subtract(W, step_size*dw)
        b = np.subtract(b, step_size*db)

        # print('W: ', W)
      scores = np.dot(x_train, W) + b
      # print("scocre size", scores.shape)
      # print("W size", W.shape)
      predicted_class = np.argmax(scores, axis=1)
      if epoch % 5 == 0: 
          print('loss: ', loss)
          print ('training accuracy: %.4f' % (np.mean(predicted_class == y_train)))
      if ((np.mean(predicted_class == y_train)) < lastAcc):
          lastAcc = np.mean(predicted_class == y_train)
          best_w = W
          best_b = b

    testScores = np.dot(x_test, best_w) + best_b[:x_test.shape[0], :];
    predicted_class = np.argmax(testScores, axis=1)

    return best_w, best_b, np.mean(predicted_class == y_test)

best_w, best_b, acc_rate = get_result(x_train, y_train, x_test, y_test)

print('test accuracy: %.4f' % acc_rate)