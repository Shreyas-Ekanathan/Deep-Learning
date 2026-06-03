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
        self.h1_state = np.zeros((self.batch_size, 256)) #an activation of length 256 per batch
        
        #output batch_size x 144
        self.w2_self = np.random.randn(144, 144) * np.sqrt(2.0 / 288)
        self.w2_input = np.random.rand(256, 144) * np.sqrt(2.0 / 400)
        self.b2 = np.zeros(144)
        self.h2_state = np.zeros((self.batch_size, 144)) #activation of length 144 per batch
        
        #output batch_size x 95
        self.w_out = np.random.randn(144, 95) * np.sqrt(2.0 / 239)
        self.b_out = np.zeros(95)
        
        #adam stuff
        self.w1_self_m = np.zeros((256, 256))
        self.w1_self_v = np.zeros((256, 256))
        self.w1_input_m = np.zeros((95, 256))
        self.w1_input_v = np.zeros((95, 256))
        self.b1_m = np.zeros(256)
        self.b1_v = np.zeros(256)
        
        self.w2_self_m = np.zeros((144, 144))
        self.w2_self_v = np.zeros((144, 144))
        self.w2_input_m = np.zeros((256, 144))
        self.w2_input_v = np.zeros((256, 144))
        self.b2_m = np.zeros(144)
        self.b2_v = np.zeros(144)
        
        self.w_out_m = np.zeros((144, 95))
        self.w_out_m = np.zeros((144, 95))
        self.b_out_m = np.zeros(95)
        self.b_out_v = np.zeros(95)
        
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
        z1 = self.h1_state @ self.w1_self + batch @ self.w1_input+ self.b1 #batch_size x 256
        self.h1_state = np.tanh(z1) #batch_size x 256
        z2 = self.h2_state @ self.w2_self + self.h1_state @ self.w2_input + self.b2 #shape batch_size x 144
        self.h2_state = np.tanh(z2) # batch_size x 144
        z3 = self.h2_state @ self.w_out + self.b_out #y3 has shape batch_size x 95
        prediction = self.softmax(z3)
        return z1, z2, z3, prediction
    
    def backward_pass(self, prediction, target, z2, z1, batch, l2_hidden_states, l1_hidden_states):
        batch_size = self.batch_size
        z3_bar = (prediction - target) / batch_size #batch_size x 95
        w_out_bar = self.h2_state.T @ z3_bar + reg_coef * self.w_out #144 x 95
        b_out_bar = np.sum(z3_bar, axis = 0) + self.b_out #95x1
        #we are going to do BPTT over say the past 100 characters
        #need to accumulate the gradient
        
        z2_bar = np.zeros((batch_size, 144))
        h2_state_bar = np.zeros((batch_size, 144))
        w2_self_bar = np.zeros((144, 144))
        w2_input_bar = np.zeros((256, 144))
        b2_bar = np.zeros((144))
        
        z1_bar = np.zeros((batch_size, 256))
        h1_state_bar = np.zeros((batch_size, 256))
        w1_state_bar = np.zeros((256, 256))
        w1_input_bar = np.zeros((95, 256))
        b1_bar = np.zeros((256))
        for i in reversed(range(100)):
            l2_i = l2_hidden_states[i] #batch_size x 144
            l1_i = l1_hidden_states[i] #batch_size x 256
            
            h2_state_bar = z3_bar @ self.w_out.T + z2_bar @ self.w2_self.T # batch_size x 144 
            w2_self_bar += z2_bar.T @ l2_i #144x144
            z2_bar = h2_state_bar * (1 - z2 ** 2) #batch_sizex144
            w2_input_bar += l1_i.T @ z2_bar #256x144
            b2_bar += np.sum(z2_bar, axis=0) #144x1
            
            h1_state_bar = z2_bar @ self.w2_input.T + z1_bar @ self.w1_self.T #batch_size x 256
            w1_self_bar += z1_bar.T @ l1_i #256x256
            z1_bar = h1_state_bar * (1 - z1 ** 2) #batch_size x 256
            w1_input_bar += batch.T @ z1_bar #95x256
            b1_bar += np.sum(z1_bar, axis=0)

        w2_self_bar += self.w2_self * reg_coef
        w2_input_bar += self.w2_input * reg_coef
        b2_bar += self.b2 * reg_coef
        w1_self_bar += self.w1_self * reg_coef
        w1_input_bar += self.w1_input * reg_coef
        b1_bar += self.b1 * reg_coef
        
        return b1_bar, w1_input_bar, w1_self_bar, b2_bar, w2_input_bar, w2_self_bar, w_out_bar, b_out_bar
        
        
    def adam(self, prediction, target, z2, z1, batch, l2_hidden_states, l1_hidden_states):
        (b1_bar, w1_input_bar, w1_self_bar, b2_bar, 
         w2_input_bar, w2_self_bar, w_out_bar, b_out_bar) = self.backprop(prediction, target, z2, z1, batch, l2_hidden_states, l1_hidden_states)
        
        #m updates
        self.w1_self_m = self.w1_self_m * beta + (1 - beta) * w1_self_bar
        self.w1_input_m = self.w1_input_m * beta + (1 - beta) * w1_input_bar
        self.b1_m = self.b1_m * beta + (1 - beta) * b1_bar
        self.w2_self_m = self.w2_self_m * beta + (1 - beta) * w2_self_bar
        self.w2_input_m = self.w2_input_m * beta + (1 - beta) * w2_input_bar
        self.b2_m = self.b2_m * beta + (1 - beta) * b2_bar
        self.w_out_m = self.w_out_m * beta + (1 - beta) * w_out_bar
        self.b_out_m = self.b_out_m * beta + (1 - beta) * b_out_bar

        #v updates
        self.w1_self_v = self.w1_self_v * gamma + (1 - gamma) * (w1_self_bar ** 2)
        self.w1_input_v = self.w1_input_v * gamma + (1 - gamma) * (w1_input_bar ** 2)
        self.b1_v = self.b1_v * gamma + (1 - gamma) * (b1_bar ** 2)
        self.w2_self_v = self.w2_self_v * gamma + (1 - gamma) * (w2_self_bar ** 2)
        self.w2_input_v = self.w2_input_v * gamma + (1 - gamma) * (w2_input_bar ** 2)
        self.b2_v = self.b2_v * gamma + (1 - gamma) * (b2_bar ** 2)
        self.w_out_v = self.w_out_v * gamma + (1 - gamma) * (w_out_bar ** 2)
        self.b_out_v = self.b_out_v * gamma + (1 - gamma) * (b_out_bar ** 2)
        
        #param updates
        self.w1_self -= alpha * self.w1_self_m / (np.sqrt(self.w1_self_v) + 1e-12)
        self.w1_input -= alpha * self.w1_input_m / (np.sqrt(self.w1_input_v) + 1e-12)
        self.b1 -= alpha * self.b1_m / (np.sqrt(self.b1_v) + 1e-12)
        self.w2_self -= alpha * self.w2_self_m / (np.sqrt(self.w2_self_v) + 1e-12)
        self.w2_input -= alpha * self.w2_input_m / (np.sqrt(self.w2_input_v) + 1e-12)
        self.b2 -= alpha * self.b2_m / (np.sqrt(self.b2_v) + 1e-12)
        self.w_out -= alpha * self.w_out_m / (np.sqrt(self.w_out_v) + 1e-12)
        self.b_out -= alpha * self.b_out_m / (np.sqrt(self.b_out_v) + 1e-12)
        
reg_coef = 0.075
beta = 0.9
gamma = 0.99
alpha = 0.001
