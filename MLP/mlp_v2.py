#what will we build on?
# 1. Add a regularizer and see how that changes the performance of the model (ridge)
# 2. stochastic gradient descent + momentum instead of normal gradient descent (ADAM)
# 3. we'll do a classifier instead of a regression model, to explore other activation functions (e.g. softmax)
# 4. dropout and batch normalization (normalize outputs of one layer as they go into the next)
# 5. cross entropy loss instead of MSE

# we will train a classifier to learn some sort of strange spiral shape or something like that
# model architecture:
# 3 layers
# first layer takes input -> batch normalize -> ReLU -> second layer -> batch normalize -> ReLU -> third layer -> softmax -> output
# training: on any given iteration, we'll kill say 30% of neurons. apply ADAM and batching to train.
# loss function will have a L2 regularizer 

import numpy as np

class model: 
    def __init__(self):
        self.w1 = np.zeros((16, 128))
        self.b1 = np.zeros(128)
        self.w2 = np.zeros((128, 32))
        self.b2 = 32
        self.w3 = np.zeros(32)
        self.b3 = 0
        
    def cross_entropy_loss(prediction, target, batch_size):
        return np.sum(-target[i] * np.log(prediction[i]) for i in range(batch_size))
    
    def regularized_loss(self, prediction, target, batch_size, reg_coef):
        return self.cross_entropy_loss(prediction, target, batch_size) + reg_coef * (self.w1 ** 2 + self.w2 ** 2 + self.w3 ** 2 + self.b1 ** 2 + self.b2 ** 2 + self.b3 ** 2) 