import numpy as np

# This one covers multiple linear regression as well.
# 1. Initiate slope and intercept as 0 initially
# 2. Iterate through steps 3
# 3. Calculate y as y_pred with current slope and intercept
# 4. Calculate slope and bias variation as
#   - slope = Mean of Sum of Xi * (Ypred - pred)
#   - Bias = Mean difference of current predicted vs actual values
# 5. Update slope and bias by removing/(or adding based on eqn) to existing values by the factor of learning rate
# 6. Update learning rate and iteration by trial and error (lr : 0.5 0.1 0.01... ,iteration: 500,1000,2000... )


class LinearRegression:
    def __init__(self) -> None:
        self.weights = None
        self.bias = None
        self.theta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        learning_rate = 0.05
        n_iteration = 4000

        for i in range(n_iteration):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            dw = (1/n_samples) * np.dot(X.T, (error))
            db = (1/n_samples) * np.sum(error)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return (np.dot(X, self.weights) + self.bias)

    def stats(self):
        print("weights : ", np.round(self.weights, decimals=3))
        print("bias : ", np.round(self.bias, decimals=3))
