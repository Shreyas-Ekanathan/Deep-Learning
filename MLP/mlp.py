import numpy as np
#lets make a model with an input layer, 1 hidden layer, and one output layer
#we'll model the data off of some nonlinear thing, e.g. lets have the model learn a polynomial with sinusoidal stuff

#input layer dim 64
#hidden layer dim 256
#output layer dim 32
#let's use relu to activate each just for demonstration
# we will also batch inputs for demonstration, take a random batch repeatedly and then also store a test batch to compare against

#result: model learns shannon entropy!
def generate_data(training_size):
    training_set_in = np.zeros((training_size, 64))
    test_set_in = np.zeros((1000, 64))
    training_set_out = np.zeros(training_size)
    test_set_out = np.zeros(1000)
    for i in range(training_size):
        x = np.random.uniform(0.0, 1.0, 64)
        y = np.sum(x * np.log(x))
        training_set_in[i] = x
        training_set_out[i] = y
        
    for i in range(1000):
        x = np.random.uniform(0.0, 1.0, 64)
        y = np.sum(x * np.log(x))
        test_set_in[i] = x
        test_set_out[i] = y
    return training_set_in, training_set_out, test_set_in, test_set_out

def get_batch(input, output, batch_size):
    indices = [np.random.randint(1, 10000 - batch_size) for _ in range (batch_size)]
    batch_input = np.zeros((batch_size, 64))
    batch_output = np.zeros(batch_size)
    for i in range(batch_size):
        batch_input[i] = input[indices[i]]
        batch_output[i] = output[indices[i]]
    return batch_input, batch_output

batch_size = 1000

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
        out1 = X @ self.hidden1 + self.b1 #output is 100x256, this is y1
        activated1 = self.relu(out1) #z1, 100x256
        out2 = activated1 @ self.hidden2 + self.b2 #output is 100x32, y2
        activated2 = self.relu(out2) # z2, 100x32
        y = activated2 @ self.hidden3 + self.b3 #100x1, 1 output per each of the hundred inputs, this is out
        return out1, activated1, out2, activated2, y
    
    def MSE_loss(self, prediction, target, size):
        return 0.5 / size * np.linalg.norm(target - prediction) ** 2
    
    def backprop_updates(self, x, y1, z1, y2, z2, prediction, target, learning_rate):
        p_bar = (prediction - target) / batch_size #100x1
        w3_bar = z2.T @ p_bar #32x1
        b3_bar = np.sum(p_bar, axis = 0) #1x1
        z2_bar = p_bar[:, None] * self.hidden3[None, :] #100x1 * 1x32 -> 100x32
        y2_bar = z2_bar * (y2 > 0) #100x32
        w2_bar = z1.T @ y2_bar #should be 256x32, is 256x100 * 100x32
        b2_bar = np.sum(y2_bar, axis = 0) #32x1
        z1_bar = y2_bar @ self.hidden2.T # 100x32 * 32x256 -> 100x256
        y1_bar = z1_bar * (y1 > 0) #100x256
        w1_bar = x.T @ y1_bar #64x100 * 100x256 -> 64x256
        b1_bar = np.sum(y1_bar, axis = 0) #256x1
        
        self.hidden3 = self.hidden3 - learning_rate * w3_bar
        self.b3 = self.b3 - learning_rate * b3_bar
        self.hidden2 = self.hidden2 - learning_rate * w2_bar
        self.b2 = self.b2 - learning_rate * b2_bar
        self.hidden1 = self.hidden1 - learning_rate * w1_bar
        self.b1 = self.b1 - learning_rate * b1_bar
         

train_set_input, training_set_output, test_set_input, test_set_output = generate_data(1000000)
m = model()

learning_rate = 0.1
for i in range(1000): #1000 training epochs
    batch, target = get_batch(train_set_input, training_set_output, batch_size)
    y1, z1, y2, z2, prediction = m.forward_pass(batch)
    m.backprop_updates(batch, y1, z1, y2, z2, prediction, target, learning_rate)
    if (i % 50 == 0):
        #evaluate on test set
        y1, z1, y2, z2, prediction = m.forward_pass(test_set_input)
        print("loss: ", m.MSE_loss(prediction, test_set_output, 1000))