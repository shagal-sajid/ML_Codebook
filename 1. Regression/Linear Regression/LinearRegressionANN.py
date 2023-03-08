import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000, hidden_layer_size=5):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.W1 = None
        self.W2 = None

    def fit(self, X, y):
        W1 = np.random.randn(X.shape[1], self.hidden_layer_size)
        W2 = np.random.randn(self.hidden_layer_size, 1)

        for i in range(self.num_epochs):
            Z1 = np.dot(X, W1)
            A1 = np.maximum(Z1, 0)
            y_pred = np.dot(A1, W2)
            y_pred = y_pred.reshape((y_pred.shape[0],))

            loss = np.mean(np.square(y_pred - y))
            print("epoch: ", i, "loss : ", loss)

            err = y_pred - y
            err = err.reshape((err.shape[0], 1))

            dW2 = np.dot(A1.T, err) / len(y)
            dA1 = np.dot(err, W2.T)
            dZ1 = dA1.copy()
            dZ1[Z1 < 0] = 0
            dW1 = np.dot(X.T, dZ1) / len(y)

            W2 -= self.learning_rate * dW2
            W1 -= self.learning_rate * dW1

        self.W1 = W1
        self.W2 = W2

    def stats(self):
        print("printing W1 metrix, Shape : ", self.W1.shape)
        print(self.W1)
        print("printing W2 metrix, Shape : ", self.W2.shape)
        print(self.W2)
