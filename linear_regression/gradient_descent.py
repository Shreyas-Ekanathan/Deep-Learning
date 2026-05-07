#does gradient descent 
import numpy as np

def generate_data():
    slopes = np.random.uniform(-1.0, 1.0, 64)
    intercept = np.random.uniform(-1.0, 1.0)
    x_vals = np.zeros((10000, 64))
    y_vals = np.zeros(10000)
    for i in range(10000):
        x = np.random.uniform(-1.0, 1.0, 64) #this is why we normalize! gradients blew up with bigger x values
        y = np.dot(x, slopes) + intercept + np.random.uniform(-0.1, 0.1)
        x_vals[i] = x
        y_vals[i] = y
    return x_vals, y_vals, slopes, intercept

data_x, data_y, true_slope, true_intercept = generate_data()
#append a 1 as the last features of each data_x to avoid biases
augmented_x = np.zeros((10000, 65))
for i in range(10000):
    augmented_x[i] = np.append(data_x[i], 1)

xT = augmented_x.T

class model:
    def __init__(self):
        self.weights = np.zeros(65)
        
m = model()

for i in range(1000): #1000 training epochs
    y = augmented_x @ m.weights
    residual = y - data_y
    update = 1/10000 * (xT @ residual)
    m.weights = m.weights - update
    
weights_post_training = m.weights
true_weights = np.append(true_slope, true_intercept)
print(np.linalg.norm(weights_post_training - true_weights, ord=None, axis=None, keepdims=False))
print(weights_post_training)
print(true_weights)