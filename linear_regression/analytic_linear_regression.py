#this file will use linear algebra to analytically solve for the best fit line of a set of data points
import numpy as np

def generate_data():
    slopes = np.random.uniform(-204004.96, 1204021490124.44, 64)
    intercept = np.random.uniform(-12049024.24, 1409204.24)
    x_vals = np.zeros((10000, 64))
    y_vals = np.zeros(10000)
    for i in range(10000):
        x = np.random.uniform(-12049109402.4, 04192091049.4, 64)
        y = np.dot(x, slopes) + intercept + np.random.uniform(-0.0001, 0.00001)
        x_vals[i] = x
        y_vals[i] = y
    return x_vals, y_vals, slopes, intercept

data_x, data_y, true_slope, true_intercept = generate_data()
#append a 1 as the last features of each data_x to avoid biases
augmented_x = np.zeros((10000, 65))
for i in range(10000):
    augmented_x[i] = np.append(data_x[i], 1)
    
xT = augmented_x.T
A = xT @ augmented_x
A_inv = np.linalg.inv(A)
b = xT @ data_y
weights = A_inv @ b
true_weights = np.append(true_slope, true_intercept)
print(np.linalg.norm(weights-true_weights, ord=None, axis=None, keepdims=False))