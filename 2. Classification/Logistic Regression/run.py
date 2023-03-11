import numpy as np
from LogisticRegression import LogisticRegression


def sigmoid(x):
    return np.round(1 / (1 + np.exp(-x)),  decimals=0)


X = np.random.rand(100, 2)
z = np.dot(X, [1, -1]) + 0.01
y = sigmoid(z)

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
model.stats(y, y_pred)
