import numpy as np


def generate_logistic_data(num_samples, num_features, bias=0.5, scale=1):
    X = np.random.normal(size=(num_samples, num_features))
    w = np.random.normal(size=(num_features, 1))
    b = bias
    y = 1 / (1 + np.exp(-(np.dot(X, w) + b)))
    y = np.where(y >= 0.5, 1, 0)
    X = X * scale
    X = np.round(X, decimals=4)
    return X, y


# Example usage:
X, y = generate_logistic_data(
    num_samples=10, num_features=5, bias=0.5, scale=10)
print(X)
print(y)
