# cnn architecture
# lets train a model to work on MNIST
from torchvision import datasets, transforms
import numpy as np

train_data = datasets.MNIST(root='./CNN', train=True, download=True, transform=transforms.ToTensor())
test_data  = datasets.MNIST(root='./CNN', train=False, download=True, transform=transforms.ToTensor())

x_train = train_data.data.numpy().reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_data.targets.numpy()

x_test = test_data.data.numpy().reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = test_data.targets.numpy()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class model:
    def __init__(self):
        #need to do He init later
        self.kernel1 = np.zeros((9, 9, 1, 18)) #18 9x9 filters
        self.k1_m = np.zeros((9, 9, 1, 18)) #adam stuff
        self.k1_v = np.zeros((9, 9, 1, 18))
        self.kernel_b1 = np.zeros(18) #one val per filter
        self.k_b1_m = np.zeros(18)
        self.k_b1_v = np.zeros(18)
        
        #dims are height, width, input_channels, output_channels
        #we will max pool between these two layers
        #output for a given image is 28x28x18, pooling makes it 14x14x18
        self.kernel2 = np.zeros((5, 5, 18, 9)) #9 5x5 filters
        self.k2_m = np.zeros((5, 5, 18, 9))
        self.k2_v = np.zeros((5, 5, 18, 9))
        self.kernel_b2 = np.zeros(9)
        self.k_b2_m = np.zeros(9)
        self.k_b2_v = np.zeros(9)
        
        #another max pool
        #output for a given image is 14x14x9, pooling makes it 7x7x9
        #after flattening, its 441
        self.fc1 = np.zeros((441, 128))
        self.fc1_m = np.zeros((441, 128))
        self.fc1_v = np.zeros((441, 128))
        
        self.fc_b1 = np.zeros(128)
        self.fc_b1_m = np.zeros(128)
        self.fc_b1_v = np.zeros(128)
        
        self.fc2 = np.zeros((128, 10)) #this gives output
        self.fc2_m = np.zeros((128, 10))
        self.fc2_v = np.zeros((128, 10))
        
        self.fc_b2 = np.zeros(10)
        self.fc_b2_m = np.zeros(10)
        self.fc_b2_v = np.zeros(10)
        
    def cross_entropy_loss(self, prediction, target):
        return np.mean(np.sum(-target * np.log(prediction + 1e-12), axis = 1))
    
    def regularized_loss(self, prediction, target):
        return (self.cross_entropy_loss(prediction, target) 
                + reg_coef * (np.sum(self.kernel1 ** 2) + np.sum(self.kernel_b1 ** 2) + np.sum(self.kernel2 ** 2) + np.sum(self.kernel_b2 ** 2)
                              + np.sum(self.fc1 ** 2) + np.sum(self.fc_b1 ** 2) + np.sum(self.fc2 ** 2) + np.sum(self.fc_b2 ** 2)))
        
    #TODO: implement batchnorm
    
    def relu(self, input):
        return np.maximum(0, input)
    
    def softmax(self, y):
        y_shifted = y - np.max(y, axis=-1, keepdims=True)
        exp_y = np.exp(y_shifted)
        return exp_y / np.sum(exp_y, axis=-1, keepdims=True)
        
    def forward_pass(self, batch):
        #todo: once batch norm is done, integrate it into here
        
        #for efficient matrix multiplication, we need to do some funny business with the batch itself.
        #batch has shape batch_size x 28 x 28 x 1
        # we need to first pad the batch to make it batch_size x 32 x 32 x 1
        # first filter has shape 9 x 9 x 1 x 18
        pad = 4
        padded = np.pad(batch, ((0,0), (pad, pad), (pad, pad), (0,0)), mode='constant', constant_values=0)
        #im2col: take the input image and convert it into a form for convolution to be a matrix multiplication
        #allocate a new array of size batch_size x 784 (28^2) x 81 for the batch
        # and a new array of size 81 x 18 for the filter. this part is easy? just rows
        filter = self.kernel1.reshape(-1, 18) #81 x 18
        flattened1 = np.zeros((batch_size, 784, 81))
        for i in range(28):
            for j in range(28):
                for n in range(batch_size):
                    subset = padded[n, i : i + 9, j : j + 9, 0]
                    subset = subset.reshape(81)
                    flattened1[n, 28 * i + j] = subset

        y1 = flattened1 @ filter + self.kernel_b1
        z1 = self.relu(y1) #batch_size x 784 x 18
        
        # now we need to pool. this first requires unrolling z1
        # unroll to dims batch_size x 28 x 28 x 18
        unrolled1 = z1.reshape(batch_size, 28, 28, 18)
        # now we pool. output shape will be batch_size x 14 x 14 x 18
        pooled1 = np.zeros(batch_size, 14, 14, 18)
        for n in range(batch_size):
            for m in range(18):
                for i in range(14):
                    for j in range(14):
                        #4 indices: 2i, 2i+1 for x, 2j, 2j+1 for y
                        max = np.maximum(unrolled1[n, 2 * i, 2 * j, m], unrolled1[n, 2 * i + 1, 2 * j, m], 
                                         unrolled1[n, 2 * i, 2 * j + 1, m], unrolled1[n, 2 * i + 1, 2 * j + 1, m])
                        pooled1[n, i, j, m] = max
        
        # now the second convolution. pooled has shape batch_size x 14 x 14 x 18
        # the second filter has shape 5 x 5 x 18 x 9
        filter2 = self.kernel2.reshape(-1, 9) # 450 x 9
        pad2 = 2
        padded2 = np.pad(batch, ((0,0), (pad2, pad2), (pad2, pad2), (0,0)), mode='constant', constant_values=0)
        #shape: batch_size x 15 x 15 x 18
        #extract 5x5x18 patches
        flattened2 = np.zeros((batch_size, 196, 450))
        for n in range(batch_size):
            for i in range(14):
                for j in range(14):
                    flattened2[n, i * 14 + j] = padded2[n, i : i + 3, j : j + 3, :].reshape(-1)
                    
        y2 = flattened2 @ filter2 + self.kernel_b2 #196 x 9
        z2 = self.relu(y2) #196 x 9
        
        #unroll and pool again
        unrolled2 = z2.reshape(batch_size, 14, 14, 9)
        pooled2 = np.zeros(batch_size, 7, 7, 9)
        for n in range(batch_size):
            for m in range(9):
                for i in range(7):
                    for j in range(7):
                        #4 indices: 2i, 2i+1 for x, 2j, 2j+1 for y
                        max = np.maximum(unrolled2[n, 2 * i, 2 * j, m], unrolled2[n, 2 * i + 1, 2 * j, m], 
                                         unrolled2[n, 2 * i, 2 * j + 1, m], unrolled2[n, 2 * i + 1, 2 * j + 1, m])
                        pooled2[n, i, j, m] = max

        #flatten into 1 layer for the fully connected layers
        flattened3 = pooled2.reshape(batch_size, -1) #batch_size x 441
        y3 = flattened3 @ self.fc1 + self.fc_b1 #batch_size x 128
        z3 = self.relu(y3) #batch_size x 128
        
        y4 = z3 @ self.fc2 + self.fc_b2 #batch_size x 10
        prediction = self.softmax(y4) #batch_size x 10
        
    def backprop(self, prediction, target, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2):
        y4_bar = (prediction - target) / batch_size #batch_size x 10
        z3_bar = y4_bar @ self.fc2.T #batch_size x 128
        fc2_bar = z3.T @ y4_bar + reg_coef * self.fc2 #128x10
        fc_b2_bar = np.sum(y4_bar, axis = 0) + reg_coef * self.fc_b2 #10x1
        
        y3_bar = (y3 > 0) * z3_bar # batch_size x 128
        flattened3_bar = y3_bar @ self.fc1.T #batch_size x 441
        fc1_bar = flattened3.T @ y3_bar + reg_coef * self.fc1 #441x128
        fc_b1_bar = np.sum(y3_bar, axis = 0) + reg_coef * self.fc_b1 #128x1
        
        pooled2_bar = flattened3_bar.reshape(batch_size, 7, 7, 9)
        unrolled2_bar = np.zeros(batch_size, 14, 14, 9)
        for n in range(batch_size):
            for m in range(9):
                for i in range(7):
                    for j in range(7):
                        #2i, 2i+1, 2j, 2j+1
                        if (unrolled2[n, 2*i, 2*j, m] == pooled2[n, i, j, m]):
                            unrolled2_bar[n, 2*i, 2*j, m] = pooled2_bar[n, i, j, m]
                        elif (unrolled2[n, 2*i, 2*j+1, m] == pooled2[n, i, j, m]):
                            unrolled2_bar[n, 2*i, 2*j+1, m] = pooled2_bar[n, i, j, m]
                        elif (unrolled2[n, 2*i+1, 2*j, m] == pooled2[n, i, j, m]):
                            unrolled2_bar[n, 2*i+1, 2*j, m] = pooled2_bar[n, i, j, m]
                        elif (unrolled2[n, 2*i+1, 2*j+1, m] == pooled2[n, i, j, m]):
                            unrolled2_bar[n, 2*i+1, 2*j+1, m] = pooled2_bar[n, i, j, m]
        
        z2_bar = unrolled2_bar.reshape(batch_size, 196, 9) 
        y2_bar = (y2 > 1) * z2_bar #batch_size x 196 x 9
        flattened2_bar = y2_bar @ filter2.T #batch_size x 196 x 450
        filter2_bar = np.einsum('bpk,bpj->kj', flattened2, z2_bar)  # 450x9
        kernel_b2_bar = np.sum(y2_bar, axis=0) + reg_coef * self.kernel_b2
        return 0
    
    def adam():
        return 0
        
        
        
batch_size = 100
reg_coef = 0.075