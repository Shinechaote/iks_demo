import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.activations import linear, relu, sigmoid
import logging
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

class MNISTDataLoader(object):
    def __init__(self, dataFilePath, labelFilePath):
        self.dataFilePath = dataFilePath
        self.labelFilePath = labelFilePath
        self.data = []
        self.labels = []
        self.numImages = 0
    
    def readData(self):
        with open(self.labelFilePath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            self.numImages = size
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}, likely due to incorrect file name or corruption'.format(magic))
            self.labels = np.array(array("B", file.read()))
        
        with open(self.dataFilePath, 'rb') as file:
            magic, size, rows, columns = struct.unpack(">IIII", file.read(16))
            if magic != 2051 or size != self.numImages or rows != 28 or columns != 28:
                raise ValueError("Error, data is incorrect in the datafilepath")
            self.data = np.array(array("B", file.read()), dtype=np.float32)/255
            self.data = self.data.reshape(self.numImages, 784)
    def VisualiseSample(self, sampleNumber):
        images = np.array(self.data).reshape(self.numImages, 28,28)
        plt.imshow(images[sampleNumber], cmap="gray")
        plt.show()

trainingLoader = MNISTDataLoader("SampleDataset/Train/train-images-idx3-ubyte", "SampleDataset/Train/train-labels-idx1-ubyte")
trainingLoader.readData()
testLoader = MNISTDataLoader("SampleDataset/Test/t10k-images-idx3-ubyte", "SampleDataset/Test/t10k-labels-idx1-ubyte")
testLoader.readData()
trainingLoader.VisualiseSample(7231)




