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

def generate_data():
    #TODO
    return 0

def get_batch(input, output, batch_size):
    indices = [np.random.randint(1, 10000 - batch_size) for _ in range (batch_size)]
    batch_input = np.zeros((batch_size, 64))
    batch_output = np.zeros(batch_size)
    for i in range(batch_size):
        batch_input[i] = input[indices[i]]
        batch_output[i] = output[indices[i]]
    return batch_input, batch_output

class model: 
    def __init__(self):
        #TODO: implement an actual initialization scheme
        self.w1 = np.zeros((16, 128))
        self.w1_m = np.zeros((16, 128))
        self.w1_v = np.zeros((16, 128))
        
        self.b1 = np.zeros(128)
        self.b1_m = np.zeros(128)
        self.b1_v = np.zeros(128)
        
        self.w2 = np.zeros((128, 32))
        self.w2_m = np.zeros((128, 32))
        self.w2_v = np.zeros((128, 32))
        
        self.b2 = np.zeros(32)
        self.b2_m = np.zeros(32)
        self.b2_v = np.zeros(32)
        
        self.w3 = np.zeros((32, 10))
        self.w3_m = np.zeros((32, 10))
        self.w3_v = np.zeros((32, 10))
        
        self.b3 = np.zeros(10)
        self.b3_m = np.zeros(10)
        self.b3_v = np.zeros(10)
        
        self.gamma1 = np.ones(128) 
        self.gamma2 = np.ones(32)
        self.beta1 = 0
        self.beta2 = 0
        
    def cross_entropy_loss(prediction, target):
        return np.mean(np.sum(-target * np.log(prediction + 1e-12), axis = 1))
    
    def regularized_loss(self, prediction, target, reg_coef):
        return self.cross_entropy_loss(prediction, target, batch_size) + reg_coef * (self.w1 ** 2 + self.w2 ** 2 + self.w3 ** 2 + self.b1 ** 2 + self.b2 ** 2 + self.b3 ** 2) 
    
    def relu(y):
        return np.maximum(0, y)    
    
    def softmax(y):
        y_shifted = y - np.max(y)
        exp_y = np.exp(y_shifted)
        return exp_y / np.sum(exp_y)    
    
    def batch_normalize(y):
        mu = np.mean(y, axis = 0, keepdims = True) #1 x 32
        sigma2 = np.var(y, axis = 0, keepdims = True) #1 x 32
        y_norm = (y - mu) / np.sqrt(sigma2 + 1e-12)
        return y_norm, mu, sigma2
    
    def batch_norm_backprop(y, mu, sigma2, upstream_derivative):
        sigma_hat = np.sqrt(sigma2 + 1e-12)
        p1 = batch_size * upstream_derivative
        p2 = np.sum(upstream_derivative, axis = 0, keepdims = True)
        p3 = y * np.sum(upstream_derivative * y, axis = 0, keepdims = True)
        return (p1 + p2 + p3) / (batch_size * sigma_hat) #we really need to learn matrix calc properly

    def forward_pass(self, input_batch):
        y1 = input_batch @ self.w1 + self.b1 #batch_size x 128
        norm1, mu1, sigma2_1 = self.batch_normalize(y1)
        norm1_scaled = self.gamma1 * norm1 + self.beta1 #batch_size x 128
        z1 = self.relu(norm1_scaled) # batch_size x 128
        y2 = z1 @ self.w2 + self.b2 # batch_size x 32
        norm2, mu2, sigma2_2 = self.batch_normalize(y2)
        norm2_scaled = self.gamma2 * norm2 + self.beta2
        z2 = self.relu(norm2_scaled) # batch_size x 32
        y3 = z2 @ self.w3 + self.b3 #batch_size x 10 <- one hot encoding, each row is the predictions for one sample
        prediction = self.softmax(y3)
        return y1, norm1, mu1, sigma2_1, norm1_scaled, z1, y2, norm2, mu2, sigma2_2, norm2_scaled, z2, y3, prediction
            
    def backprop(self, input_batch, y1, norm1, mu1, sigma2_1, norm1_scaled, z1, 
                 y2, norm2, mu2, sigma2_2, norm2_scaled, z2, y3, prediction, target):
        y3_bar = (prediction - target) / batch_size #batch_size x 10
        w3_bar = z2.T @ y3_bar #32xbatch_size * batch_size x 10 -> 32x10
        b3_bar = np.sum(y3_bar, axis = 0)
        w3_v_update = w3_bar ** 2
        b3_v_update = b3_bar ** 2
        z2_bar = y3_bar @ self.w3.T # batch_sizex10 * 10x32 -> batch_size x 32
        norm2_scaled_bar = z2_bar * (norm2_scaled > 0)
        gamma2_bar = np.sum(norm2 * norm2_scaled_bar, axis = 0) # vector of length 32
        beta2_bar = np.sum(norm2_scaled_bar, axis = 0) #len 32
        norm2_bar = self.gamma2 * norm2_scaled_bar #batch_size x 32
        y2_bar = self.batch_norm_backprop(norm2, mu2, sigma2_2, norm2_bar) #batch_size x 32
        w2_bar = z1.T @ y2_bar #128x32
        b2_bar = np.sum(y2_bar, axis = 0) #1x32
        w2_v_update = w2_bar ** 2
        b2_v_update = b2_bar ** 2
        z1_bar = y2_bar @ self.w2.T #batch_size x 128
        norm1_scaled_bar = z1_bar * (norm1_scaled > 0) #batch_size x 128
        gamma1_bar = np.sum(norm1 * norm1_scaled_bar, axis = 0) #1x128
        beta1_bar = np.sum(norm2_scaled_bar, axis = 0) #1x128
        norm1_bar = self.gamma1 * norm1_scaled_bar
        y1_bar = self.batch_norm_backprop(norm1, mu1, sigma2_1, norm1_bar)
        w1_bar = input_batch.T @ y1_bar
        b1_bar = np.sum(y1_bar, axis = 0) 
        w1_v_update = w1_bar ** 2
        b1_v_update = b1_bar ** 2

        return (w1_bar, b1_bar, w2_bar, b2_bar, w3_bar, b3_bar, 
                w1_v_update, b1_v_update, w2_v_update, b2_v_update, w3_v_update, b3_v_update,
                gamma1_bar, gamma2_bar, beta1_bar, beta2_bar)
    
    def adam(self, input_batch, y1, norm1, mu1, sigma2_1, norm1_scaled, z1, y2, norm2, mu2, sigma2_2, norm2_scaled, z2, y3, prediction, target, beta, gamma, alpha):
        #set desired updates with momentum
        (w1_m_update, b1_m_update, w2_m_update, b2_m_update, w3_m_update, b3_m_update, 
         w1_v_update, b1_v_update, w2_v_update, b2_v_update, w3_v_update, b3_v_update,
         gamma1_update, gamma2_update, beta1_update, beta2_update) = self.backprop(input_batch, y1, norm1, mu1, sigma2_1, norm1_scaled, z1, y2, norm2, mu2, sigma2_2, norm2_scaled, z2, y3, prediction, target)
        self.w1_m = beta * self.w1_m + (1 - beta) * w1_m_update
        self.b1_m = beta * self.b1_m + (1 - beta) * b1_m_update
        self.w2_m = beta * self.w2_m + (1 - beta) * w2_m_update
        self.b2_m = beta * self.b2_m + (1 - beta) * b2_m_update
        self.w3_m = beta * self.w3_m + (1 - beta) * w3_m_update
        self.b3_m = beta * self.b3_m + (1 - beta) * b3_m_update
        
        self.w1_v = gamma * self.w1_v + (1 - gamma) * w1_v_update
        self.b1_v = gamma * self.b1_v + (1 - gamma) * b1_v_update
        self.w2_v = gamma * self.w2_v + (1 - gamma) * w2_v_update
        self.b2_v = gamma * self.b2_v + (1 - gamma) * b2_v_update
        self.w3_v = gamma * self.w3_v + (1 - gamma) * w3_v_update
        self.b3_v = gamma * self.b3_v + (1 - gamma) * b3_v_update

        #update
        self.w1 -= alpha * self.w1_m / (np.sqrt(self.w1_v) + 1e-12)
        self.b1 -= alpha * self.b1_m / (np.sqrt(self.b1_m) + 1e-12)
        self.w2 -= alpha * self.w2_m / (np.sqrt(self.w2_m) + 1e-12)
        self.b2 -= alpha * self.b2_m / (np.sqrt(self.b2_m) + 1e-12)
        self.w3 -= alpha * self.w3_m / (np.sqrt(self.w3_m) + 1e-12)
        self.b3 -= alpha * self.b3_m / (np.sqrt(self.b3_m) + 1e-12)
        self.gamma1 -= alpha * gamma1_update
        self.gamma2 -= alpha * gamma2_update
        self.beta1 -= alpha * beta1_update
        self.beta2 -= alpha * beta2_update

batch_size = 100
train_set_input, training_set_output, test_set_input, test_set_output = generate_data(1000000)
m = model()

learning_rate = 0.1

for i in range(1000): #1000 training epochs
    batch, target = get_batch(train_set_input, training_set_output, batch_size)
    y1, norm1, mu1, sigma2_1, norm1_scaled, z1, y2, norm2, mu2, sigma2_2, norm2_scaled, z2, y3, prediction = m.forward_pass(batch)
    m.adam(batch, y1, norm1, mu1, sigma2_1, norm1_scaled, z1, y2, norm2, mu2, sigma2_2, norm2_scaled, z2, y3, prediction, target, 0.05, 0.05)
    if (i % 50 == 0):
        #evaluate on test set
        y1, z1, y2, z2, prediction = m.forward_pass(test_set_input)
        print("loss: ", m.MSE_loss(prediction, test_set_output, 1000))