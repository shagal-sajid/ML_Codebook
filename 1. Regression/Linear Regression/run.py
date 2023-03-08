import numpy as np
from LinearRegressionANN import LinearRegression


X = np.random.rand(100, 2)
y = np.dot(X, [55, 22]) + 0.5


# create a LinearRegression instance and fit the model
model = LinearRegression()
model.fit(X, y)
model.stats()
# model.stats()
# print("Shape of data : ", X.shape)


# create a LinearRegressionV2 instance and fit the model
# m, n = X.shape
# Xn = np.hstack((np.ones((m, 1)), X))
# model = LinearRegression()
# model.fit(Xn, y)
# model.stats()
# print("Shape of data : ", X.shape)
