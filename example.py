import numpy as np
from scipy.linalg import lstsq

# Data: Height (dependent) and Weight (independent)
weight = np.array([4.5, 5.1, 6.3, 4.8])  # Independent variable
height = np.array([155, 232, 340, 185])   # Dependent variable

# Add a column of ones to the weight to account for the intercept
A = np.vstack([weight, np.ones(len(weight))]).T

# Perform the least squares fit
slope, intercept = lstsq(A, height)[0]

# Print the results
print(f"y-intercept: {intercept}")
print(f"slope: {slope}")
