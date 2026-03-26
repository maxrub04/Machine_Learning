import numpy as np
import matplotlib.pyplot as plt
from typing import *
from pprint import pp
from sklearn.model_selection import train_test_split


pointes_per_class = 50
centers = np.array([[1, 1], [2, 2]])
x = np.array([[1, 1],[2, 2]])
y = np.array([0, 1])

for i, center in enumerate(centers):
    stddev = 1.1
    points = np.random.randn(pointes_per_class, 2)*stddev + center
    x = np.append(x, points, axis=0)
    y = np.append(y, np.array([i] * pointes_per_class))

_, ax = plt.subplots()
for i in range(2):
    ax.scatter(x[y == i, 0], x[y == i, 1], label=i)

x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2)

class Perceptron:
    def __init__(self, size: int):
        self.weights = np.random.rand(size)*2-1
        self.bias = np.random.rand()

        self.lr = 0.1
        self.epochs = 10
        self.activate = lambda x: 1 if x > 0 else 0

    def _forward(self, xs: np.ndarray) -> Literal[0,1]:
        state = np.dot(self.weights, xs)
        ret = self.activate(state+self.bias)
        return ret


    def _backward(self, xs: np.ndarray, error: Literal[-1,0,1]) -> None:
        self.bias += self.lr*error

        for i in range(self.size):
            self.wights[i] += xs[i] *self.lr * error


    def train(self, x_traing):
        pass

    def eval(self,x: np.ndarray,y: np.ndarray):
        ret = np.array(list(map(self._forward, x)))
        return ret

p = Perceptron(x.shape[1])
y_pred = p.eval(x)

for y_t, y_p in zip(y, y_pred):
    print(y_t, y_p)