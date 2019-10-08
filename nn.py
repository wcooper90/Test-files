# class imported from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import pandas as pd
from tqdm import tqdm

class NeuralNetwork():

    def __init__(self, initArray):
        self.array = initArray
        self.size = len(initArray)
        # self.size = np.shape(initArray)
        self.weights1 = []
        # self.weights2 = []
        for x in range(self.size):
            self.weights1.append(np.random.random_sample())
            # self.weights2.append(np.random.random_sample())
        self.newArray = np.zeros(self.size)

    def activationFunc(self, initArray):
        return np.tanh(initArray)

    def lossFunc(self, testArray):
        loss = self.activationFunc(self.array - testArray)
        return loss

    def feedForward(self):
        self.newArray = self.activationFunc(np.dot(self.weights1, self.newArray))
        # testArray = np.dot(self.weights2, activationFunc(testArray))
        return self.newArray

    def backPropagation(self):
        self.weights1 += self.lossFunc(self.newArray)
        return self.newArray

    def networkErrorFunction(self):
        return False


x = np.random.random_sample()
print(x)
print(np.tanh(x))

initArray = [1, 0, 1, 1]
nn = NeuralNetwork(initArray)
print(nn.array)
print(nn.weights1)
print(nn.newArray)
for x in tqdm(range(100)):
    nn.feedForward()
    nn.backPropagation()

print(nn.weights1)
print(nn.newArray)
