import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.1, iteration=200):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.theta = None

    def fit(self, X, y):

        m, n = X.shape
        self.theta = np.zeros((n, 1))

        for i in range(self.iteration):
            h = X.dot(self.theta)

            gradient = X.T.dot(h - y) / m

            self.theta -= self.learning_rate * gradient

    def predict(self, X, y):
        return X.dot(self.theta)

    def stats(self):
        print("bias : ", self.theta[0])
        print("Weights :", self.theta[1:])
