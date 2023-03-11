import numpy as np

# in practice, determining the appropriate degree of polynomial to use for a given dataset can be a complex task that requires a combination of domain knowledge, experimentation, and evaluation of the model's performance using techniques like cross-validation.


class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree

    def fit(self, x, y):

        X = np.zeros(len(x), self.degree+1)

        for i in range(len(x)):
            for j in range(self.degree+1):
                X[i][j] = x[i]**j
