import numpy as np
#lets make a model with an input layer, 1 hidden layer, and one output layer
#we'll model the data off of some nonlinear thing, e.g. lets have the model learn a polynomial with sinusoidal stuff

#input layer dim 64
#hidden layer dim 256
#output layer dim 32
#let's use relu to activate each just for demonstration
# we will also batch inputs for demonstration, take a random batch repeatedly and then also store a test batch to compare against

def generate_data():
    return 0

class model:
    def __init__(self):
        self.hidden1 = np.zeros((64, 256))
        self.b1 = np.zeros(256)
        self.hidden2 = np.zeros((256, 32))
        self.b2 = np.zeros(32)
        self.hidden3 = np.zeros(32)
        self.b3 = 0
        
    def relu(self, y):
        return np.maximum(0, y)    
    
    def forward_pass(self, X):
        out1 = X @ self.hidden1 + self.b1 #output is 100x256
        activated1 = self.relu(out1)
        out2 = activated1 @ self.hidden2 + self.b2 #output is 100x32
        activated2 = self.relu(out2) 
        y = activated2 @ self.hidden3 + self.b3 #100x1, 1 output per each of the hundrend inputs 
        return y

X = np.zeros((100, 64))


