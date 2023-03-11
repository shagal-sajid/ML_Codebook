import numpy as np

# Exactly same as linear regression, except, There will be an activation function
# Here we use sigmoid function


class LogisticRegression:
    def __init__(self) -> None:
        self.weights = None
        self.bias = None
        self.theta = None

    def sigmoid(self, x):
        return np.round(1 / (1 + np.exp(-x)),  decimals=0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        learning_rate = 0.05
        n_iteration = 4000

        for i in range(n_iteration):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            error = y_pred - y
            dw = (1/n_samples) * np.dot(X.T, (error))
            db = (1/n_samples) * np.sum(error)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        print("Done")

    def predict(self, X):
        z = (np.dot(X, self.weights) + self.bias)
        y_pred = self.sigmoid(z)
        return y_pred

    def stats(self, y, y_pred):
        n_correct = np.sum(y == y_pred)
        accuracy = (n_correct / len(y)) * 100
        print("Accuracy:", accuracy)
