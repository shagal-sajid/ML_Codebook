import numpy as np
import matplotlib.pyplot as plt

# Define the degree of the polynomial
degree = 2

# Define the range of X values
xmin, xmax = -5, 5
num_points = 50

# Generate random X values within the defined range
x = np.linspace(xmin, xmax, num_points)

# Generate random noise to add to the Y values
noise = np.random.normal(0, 20, num_points)

# Define the coefficients of the polynomial equation
beta = np.random.uniform(-10, 10, degree+1)

# Calculate the Y values for each X value using the polynomial equation
y = np.zeros(num_points)
for i in range(num_points):
    for j in range(degree+1):
        y[i] += beta[j] * x[i]**j
    y[i] += noise[i]

# Plot the data
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
