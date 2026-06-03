#what are we going to do? 
# rnn to do character prediction for text generation
#let's say we have two hidden layers for some complexity
#each will have two sets of weights: one for the hidden state and one for the input
#cross entropy loss
# let's say we have 95 chars (everything printable in ascii)
#relu activation into softmax
# one hot encodings for input 

import numpy as np

class model:
    def __init__(self):
        #input dim is of shape 95, likewise for output
        #use glorot init because activation function will be tanh
        # outputs shape of batch_size x 256
        self.batch_size = 1000
        self.w1_self = np.random.randn(256, 256) * np.sqrt(2.0 / 512)
        self.w1_input = np.random.randn(95, 256) * np.sqrt(2.0 / 351)
        self.b1 = np.zeros(256)
        self.h1_state = np.zeros((1000, 144)) #an activation of length 144 per batch
        
        #output batch_size x 144
        self.w2_self = np.random.randn(144, 144) * np.sqrt(2.0 / 288)
        self.w2_input = np.random.rand(256, 144) * np.sqrt(2.0 / 400)
        self.b2 = np.zeros(144)
        self.h2_state = np.zeros((1000, 96)) #activation of length 96 per batch
        
        #output batch_size x 95
        self.w_out = np.random.randn(144, 95) * np.sqrt(2.0 / 239)
        self.b_out = np.zeros(95)
        
    def softmax(self, y):
        y_shifted = y - np.max(y, axis=-1, keepdims=True)
        exp_y = np.exp(y_shifted)
        return exp_y / np.sum(exp_y, axis=-1, keepdims=True)
    
    def cross_entropy_loss(self, prediction, target):
        return np.mean(np.sum(-target * np.log(prediction + 1e-12), axis = 1))
    
    def regularized_loss(self, prediction, target):
        return (self.cross_entropy_loss(prediction, target) 
                + reg_coef * (np.sum(self.w1_self ** 2) + np.sum(self.w1_input ** 2) + np.sum(self.b1 ** 2)
                              + np.sum(self.w2_self ** 2) + np.sum(self.w2_input ** 2) + np.sum(self.b2 ** 2)
                              + np.sum(self.w_out ** 2) + np.sum(self.b_out ** 2)))

    def forward_pass(self, batch):
        self.h1_state = self.h1_state @ self.w1_self + batch @ self.w1_input+ self.b1 #256 x 256
        y1 = np.tanh(self.h1_state) #batch_size x 256
        self.h2_state = self.w2_self @ self.h2_state + y1 @ self.w2_input + self.b2 #shape 196x196
        y2 = np.tanh(self.h2_state) #y2 has shape batch_size x 144
        y3 = y2 @ self.w_out + self.b_out #y3 has shape batch_size x 95
        prediction = self.softmax(y3)
        return y1, y2, y3, prediction
    
    def backward_pass(self, prediction, target, y2, l2_hidden_states, l1_hidden_states):
        batch_size = prediction.shape[0]
        y3_bar = (prediction - target) / batch_size #batch_size x 95
        y2_bar = y3_bar @ self.w_out.T #batch_size x 144
        w_out_bar = y2.T @ y3_bar + reg_coef * self.w_out #144 x 95
        b_out_bar = np.sum(y3_bar, axis = 0) + self.b_out #95x1
        #we are going to do BPTT over say the past 100 characters
        #need to accumulate the gradient
        h2_state_bar = (1 - y2 ** 2) @ y2_bar # batch_size x 144 * 144x95 = batch_size x 95
        
        w2_self_bar = np.zeros((144, 144))
        w2_input_bar = np.zeros((256, 144))
        b2_bar = np.zeros((144))
        h1_state_bar = np.zeros((batch_size, 144))
        w1_state_bar = np.zeros((256, 256))
        w1_input_bar = np.zeros((95, 256))
        b1_bar = np.zeros((256))
        for i in reversed(range(100)):
            l2_i = l2_hidden_states[i]
            l1_i = l1_hidden_states[i]

        
reg_coef = 0.001