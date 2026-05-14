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
        self.w1_m1_t0 = np.zeros((16, 128))
        self.w1_v1_t0 = np.zeros((16, 128))
        self.w1_m1_t1 = np.zeros((16, 128))
        self.w1_v1_t1 = np.zeros((16, 128))
        self.b1 = np.zeros(128)
        self.b1_m1_t0 = np.zeros(128)
        self.b1_v1_t0 = np.zeros(128)
        self.b1_m1_t1 = np.zeros(128)
        self.b1_v1_t1 = np.zeros(128)
        self.w2 = np.zeros((128, 32))
        self.w2_m2_t0 = np.zeros((128, 32))
        self.w2_v2_t0 = np.zeros((128, 32))
        self.w2_m2_t1 = np.zeros((128, 32))
        self.w2_v2_t1 = np.zeros((128, 32))
        self.b2 = 32
        self.b2_m2_t0 = np.zeros(32)
        self.b2_v2_t0 = np.zeros(32)
        self.b2_m2_t1 = np.zeros(32)
        self.b2_v2_t1 = np.zeros(32)
        self.w3 = np.zeros(32)
        self.w3_m3_t0 = np.zeros(32)
        self.w3_v3_t0 = np.zeros(32)
        self.w3_m3_t1 = np.zeros(32)
        self.w3_v3_t1 = np.zeros(32)
        self.b3 = 0
        self.b3_m3_t0 = 0
        self.b3_v3_t0 = 0
        self.b3_m3_t1 = 0
        self.b3_v3_t1 = 0
        
        self.gamma1 = 0
        self.gamma2 = 0
        self.beta1 = 0
        self.beta2 = 0
        
    def cross_entropy_loss(prediction, target, batch_size):
        return np.sum(-target[i] * np.log(prediction[i]) for i in range(batch_size))
    
    def regularized_loss(self, prediction, target, batch_size, reg_coef):
        return self.cross_entropy_loss(prediction, target, batch_size) + reg_coef * (self.w1 ** 2 + self.w2 ** 2 + self.w3 ** 2 + self.b1 ** 2 + self.b2 ** 2 + self.b3 ** 2) 
    
    def relu(y):
        return np.maximum(0, y)    
    
    def softmax(y):
        #y is just a vector of numbers
        len = y.size
        sum = np.sum(np.exp(y[i]) for i in range(len))
        out = np.zeros(len)
        for i in range(len):
            out[i] = np.exp(y[i]) / sum 
        return out
    
    def batch_normalize(y):
        mu = np.mean(y, axis = 1)
        sigma = np.var(y, axis = 1)
        
        return 0
    
    def forward_pass(self, input_batch):
        y1 = input_batch @ self.w1 + self.b1 #batch_size x 128
        norm1 = self.gamma1 * self.batch_normalize(y1) + self.beta1
        z1 = self.relu(norm1) # batch_size x 128
        y2 = z1 @ self.w2 + self.b2 #batch_size x 32
        norm2 = self.gamma2 * self.batch_normalize(y2) + self.beta2
        z2 = self.relu(norm2) # batch_size x 32
        y3 = z2 @ self.w3 + self.b3 #batch_size x 1
        prediction = self.softmax(y3)
        return y1, norm1, z1, y2, norm2, z2, y3, prediction
            
    def backprop(self, input_batch, y1, ):
        return 0, 0, 0, 0, 0, 0 #TODO
    
    def adam(self, beta1, beta2, beta3, gamma1, gamma2, gamma3):
        self.w1_m1_t0 = self.w1_m1_t1
        self.b1_m1_t0 = self.b1_m1_t1
        self.w2_m2_t0 = self.w1_m1_t1
        self.b2_m2_t0 = self.b2_m2_t1
        self.w3_m3_t0 = self.w3_m3_t1
        self.b3_m3_t0 = self.b3_m3_t1
        
        self.w1_v1_t0 = self.w1_v1_t1
        self.b1_v1_t0 = self.b1_v1_t1
        self.w2_v2_t0 = self.w2_v2_t1
        self.b2_v2_t0 = self.b2_v2_t1
        self.w3_v3_t0 = self.w3_v3_t1
        self.b3_v3_t0 = self.b3_v3_t1
        
        #set desired updates with momentum
        w1_update, b1_update, w2_update, b2_update, w3_update, b3_update = self.backprop()
        self.w1_m1_t1 = beta1 * self.w1_m1_t0 + (1 - beta1) * w1_update
        self.b1_m1_t1 = beta1 * self.b1_m1_t0 + (1 - beta1) * b1_update
        self.w2_m2_t1 = beta2 * self.w2_m2_t0 + (1 - beta2) * w2_update
        self.b2_m2_t1 = beta2 * self.b2_m2_t0 + (1 - beta2) * b2_update
        self.w3_m3_t1 = beta3 * self.w3_m3_t0 + (1 - beta3) * w3_update
        self.b3_m3_t1 = beta3 * self.b3_m3_t0 + (1 - beta3) * b3_update