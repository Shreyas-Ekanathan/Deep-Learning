# cnn architecture
# lets train a model to work on MNIST
from torchvision import datasets, transforms
import numpy as np

class model:
    def __init__(self):
        self.kernel1 = np.random.randn(9, 9, 1, 18) * np.sqrt(2 / (9 * 9 * 1)) #18 9x9 filters
        self.k1_m = np.zeros((9, 9, 1, 18)) #adam stuff
        self.k1_v = np.zeros((9, 9, 1, 18))
        self.kernel_b1 = np.zeros(18) #one val per filter
        self.k_b1_m = np.zeros(18)
        self.k_b1_v = np.zeros(18)
        
        #dims are height, width, input_channels, output_channels
        #we will max pool between these two layers
        #output for a given image is 28x28x18, pooling makes it 14x14x18
        self.kernel2 = np.random.randn(5, 5, 18, 9) * np.sqrt(2 / (5 * 5 * 18)) #9 5x5 filters
        self.k2_m = np.zeros((5, 5, 18, 9))
        self.k2_v = np.zeros((5, 5, 18, 9))
        self.kernel_b2 = np.zeros(9)
        self.k_b2_m = np.zeros(9)
        self.k_b2_v = np.zeros(9)
        
        #another max pool
        #output for a given image is 14x14x9, pooling makes it 7x7x9
        #after flattening, its 441
        self.fc1 = np.random.randn(441, 128) * np.sqrt(2 / 441)
        self.fc1_m = np.zeros((441, 128))
        self.fc1_v = np.zeros((441, 128))
        
        self.fc_b1 = np.zeros(128)
        self.fc_b1_m = np.zeros(128)
        self.fc_b1_v = np.zeros(128)
        
        self.fc2 = np.random.randn(128, 10) * np.sqrt(2 / 128) #this gives output
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
        batch_size = batch.shape[0]
        pad = 4
        padded = np.pad(batch, ((0,0), (pad, pad), (pad, pad), (0,0)), mode='constant', constant_values=0)
        #im2col: take the input image and convert it into a form for convolution to be a matrix multiplication
        #allocate a new array of size batch_size x 784 (28^2) x 81 for the batch
        # and a new array of size 81 x 18 for the filter. this part is easy? just rows
        filter1 = self.kernel1.reshape(-1, 18) #81 x 18
        flattened1 = np.zeros((batch_size, 784, 81))
        for i in range(28):
            for j in range(28):
                flattened1[:, 28 * i + j] = padded[:, i : i + 9, j : j + 9, 0].reshape(batch_size, 81)

        y1 = flattened1 @ filter1 + self.kernel_b1
        z1 = self.relu(y1) #batch_size x 784 x 18
        
        # now we need to pool. this first requires unrolling z1
        # unroll to dims batch_size x 28 x 28 x 18
        unrolled1 = z1.reshape(batch_size, 28, 28, 18)
        # now we pool. output shape will be batch_size x 14 x 14 x 18
        pooled1 = np.maximum(
            np.maximum(unrolled1[:, 0::2, 0::2, :], unrolled1[:, 1::2, 0::2, :]),
            np.maximum(unrolled1[:, 0::2, 1::2, :], unrolled1[:, 1::2, 1::2, :])
        )
        
        # now the second convolution. pooled has shape batch_size x 14 x 14 x 18
        # the second filter has shape 5 x 5 x 18 x 9
        filter2 = self.kernel2.reshape(-1, 9) # 450 x 9
        pad2 = 2
        padded2 = np.pad(pooled1, ((0,0), (pad2, pad2), (pad2, pad2), (0,0)), mode='constant', constant_values=0)
        #shape: batch_size x 18 x 18 x 18
        #extract 5x5x18 patches
        flattened2 = np.zeros((batch_size, 196, 450))
        for i in range(14):
            for j in range(14):
                flattened2[:, i * 14 + j] = padded2[:, i : i + 5, j : j + 5, :].reshape(batch_size, 450)
                    
        y2 = flattened2 @ filter2 + self.kernel_b2 #196 x 9
        z2 = self.relu(y2) #196 x 9
        
        #unroll and pool again
        unrolled2 = z2.reshape(batch_size, 14, 14, 9)
        pooled2 = np.maximum(
            np.maximum(unrolled2[:, 0::2, 0::2, :], unrolled2[:, 1::2, 0::2, :]),
            np.maximum(unrolled2[:, 0::2, 1::2, :], unrolled2[:, 1::2, 1::2, :])
        )

        #flatten into 1 layer for the fully connected layers
        flattened3 = pooled2.reshape(batch_size, -1) #batch_size x 441
        y3 = flattened3 @ self.fc1 + self.fc_b1 #batch_size x 128
        z3 = self.relu(y3) #batch_size x 128
        
        y4 = z3 @ self.fc2 + self.fc_b2 #batch_size x 10
        prediction = self.softmax(y4) #batch_size x 10
        return prediction, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1
        
    def backprop(self, prediction, target, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1):
        batch_size = prediction.shape[0]
        y4_bar = (prediction - target) / batch_size #batch_size x 10
        z3_bar = y4_bar @ self.fc2.T #batch_size x 128
        fc2_bar = z3.T @ y4_bar + reg_coef * self.fc2 #128x10
        fc_b2_bar = np.sum(y4_bar, axis = 0) + reg_coef * self.fc_b2 #10x1
        
        y3_bar = (y3 > 0) * z3_bar # batch_size x 128
        flattened3_bar = y3_bar @ self.fc1.T #batch_size x 441
        fc1_bar = flattened3.T @ y3_bar + reg_coef * self.fc1 #441x128
        fc_b1_bar = np.sum(y3_bar, axis = 0) + reg_coef * self.fc_b1 #128x1
        
        pooled2_bar = flattened3_bar.reshape(batch_size, 7, 7, 9)
        unrolled2_bar = np.zeros((batch_size, 14, 14, 9))
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
        y2_bar = (y2 > 0) * z2_bar #batch_size x 196 x 9
        flattened2_bar = y2_bar @ filter2.T #batch_size x 196 x 450
        filter2_bar = np.einsum('bpk,bpj->kj', flattened2, z2_bar)  # 450x9
        kernel2_bar = filter2_bar.reshape(5, 5, 18, 9) + reg_coef * self.kernel2
        kernel_b2_bar = np.sum(y2_bar, axis=(0, 1)) + reg_coef * self.kernel_b2
        
        padded2_bar = np.zeros((batch_size, 18, 18, 18))
        for i in range(14):
            for j in range(14):
                padded2_bar[:, i:i+5, j:j+5, :] += flattened2_bar[:, i*14+j, :].reshape(batch_size, 5, 5, 18)        
                
        pooled1_bar = padded2_bar[:, 2:16, 2:16, :] #batch_size x 14 x 14 x 18
        unrolled1_bar = np.zeros((batch_size, 28, 28, 18))
        for n in range(batch_size):
            for m in range(18):
                for i in range(14):
                    for j in range(14):
                        #2i, 2i+1, 2j, 2j+1
                        if (unrolled1[n, 2*i, 2*j, m] == pooled1[n, i, j, m]):
                            unrolled1_bar[n, 2*i, 2*j, m] = pooled1_bar[n, i, j, m]
                        elif (unrolled1[n, 2*i, 2*j+1, m] == pooled1[n, i, j, m]):
                            unrolled1_bar[n, 2*i, 2*j+1, m] = pooled1_bar[n, i, j, m]
                        elif (unrolled1[n, 2*i+1, 2*j, m] == pooled1[n, i, j, m]):
                            unrolled1_bar[n, 2*i+1, 2*j, m] = pooled1_bar[n, i, j, m]
                        elif (unrolled1[n, 2*i+1, 2*j+1, m] == pooled1[n, i, j, m]):
                            unrolled1_bar[n, 2*i+1, 2*j+1, m] = pooled1_bar[n, i, j, m]

        z1_bar = unrolled1_bar.reshape(batch_size, 784, 18)
        y1_bar = (y1 > 0) * z1_bar
        filter1_bar = np.einsum('bpk,bpj->kj', flattened1, z1_bar)
        kernel1_bar = filter1_bar.reshape(9, 9, 1, 18) + reg_coef * self.kernel1
        kernel_b1_bar = np.sum(y1_bar, axis=(0, 1)) + reg_coef * self.kernel_b1
        return kernel1_bar, kernel_b1_bar, kernel2_bar, kernel_b2_bar, fc1_bar, fc_b1_bar, fc2_bar, fc_b2_bar
    
    def adam(self, prediction, target, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1):
        kernel1_bar, kernel_b1_bar, kernel2_bar, kernel_b2_bar, fc1_bar, fc_b1_bar, fc2_bar, fc_b2_bar = self.backprop(prediction, target, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1)
        
        #m updates
        self.k1_m = self.k1_m * beta + (1 - beta) * kernel1_bar
        self.k_b1_m = self.k_b1_m * beta + (1 - beta) * kernel_b1_bar
        self.k2_m = self.k2_m * beta + (1 - beta) * kernel2_bar
        self.k_b2_m = self.k_b2_m * beta + (1 - beta) * kernel_b2_bar
        self.fc1_m = self.fc1_m * beta + (1 - beta) * fc1_bar
        self.fc_b1_m = self.fc_b1_m * beta + (1 - beta) * fc_b1_bar
        self.fc2_m = self.fc2_m * beta + (1 - beta) * fc2_bar
        self.fc_b2_m = self.fc_b2_m * beta + (1 - beta) * fc_b2_bar
        
        #v updates
        self.k1_v = self.k1_v * gamma + (1 - gamma) * (kernel1_bar ** 2)
        self.k_b1_v = self.k_b1_v * gamma + (1 - gamma) * (kernel_b1_bar ** 2)
        self.k2_v = self.k2_v * gamma + (1 - gamma) * (kernel2_bar ** 2)
        self.k_b2_v = self.k_b2_v * gamma + (1 - gamma) * (kernel_b2_bar ** 2)
        self.fc1_v = self.fc1_v * gamma + (1 - gamma) * (fc1_bar ** 2)
        self.fc_b1_v = self.fc_b1_v * gamma + (1 - gamma) * (fc_b1_bar ** 2)
        self.fc2_v = self.fc2_v * gamma + (1 - gamma) * (fc2_bar ** 2)
        self.fc_b2_v = self.fc_b2_v * gamma + (1 - gamma) * (fc_b2_bar ** 2)
        
        self.kernel1 -= alpha * self.k1_m / (np.sqrt(self.k1_v) + 1e-12)
        self.kernel_b1 -= alpha * self.k_b1_m / (np.sqrt(self.k_b1_v) + 1e-12)
        self.kernel2 -= alpha * self.k2_m / (np.sqrt(self.k2_v) + 1e-12)
        self.kernel_b2 -= alpha * self.k_b2_m / (np.sqrt(self.k_b2_v) + 1e-12)
        self.fc1 -= alpha * self.fc1_m / (np.sqrt(self.fc1_v) + 1e-12)
        self.fc_b1 -= alpha * self.fc_b1_m / (np.sqrt(self.fc_b1_v) + 1e-12)
        self.fc2 -= alpha * self.fc2_m / (np.sqrt(self.fc2_v) + 1e-12)
        self.fc_b2 -= alpha * self.fc_b2_m / (np.sqrt(self.fc_b2_v) + 1e-12)
        
        
        
batch_size = 100
num_batches = 600
reg_coef = 0.075
beta = 0.9
gamma = 0.99
alpha = 0.001
N = 60000

train_data = datasets.MNIST(root='./CNN', train=True, download=True, transform=transforms.ToTensor())
test_data  = datasets.MNIST(root='./CNN', train=False, download=True, transform=transforms.ToTensor())

x_train = train_data.data.numpy().reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_raw = train_data.targets.numpy()

y_train = np.zeros((60000, 10))
for i in range(60000):
    y_train[i, y_train_raw[i]] = 1


x_test = test_data.data.numpy().reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test_raw = test_data.targets.numpy()
y_test = np.zeros((10000, 10))
for i in range(10000):
    y_test[i, y_test_raw[i]] = 1

x_test_shrunk = x_test[:1000]
y_test_shrunk = y_test[:1000]

def get_batches(input, output, batch_size, num_batches):
    batches = np.zeros((num_batches, batch_size, 28, 28, 1))
    outputs = np.zeros((num_batches, batch_size, 10))
    permutation = np.random.permutation(N)
    for i in range(num_batches):
        batch = np.zeros((batch_size, 28, 28, 1))
        out = np.zeros((batch_size, 10))
        for j in range(batch_size):
            batch[j] = input[permutation[i * batch_size + j]]
            out[j] = output[permutation[i * batch_size + j]]
        batches[i] = batch
        outputs[i] = out
    return batches, outputs

m = model()

learning_rate = 0.001
reg_coef = 0.001

for i in range(5): # 5 training epochs
    batches, targets = get_batches(x_train, y_train, batch_size, num_batches) 
    learning_rate *= 0.9
    # batch should be batch_size x 3
    for j in range(num_batches):
        batch = batches[j]
        target = targets[j]
        prediction, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1 = m.forward_pass(batch)
        m.adam(prediction, target, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1)
        if (j % 50 == 0):
            # evaluate on test set
            print("Batch ", j)
            prediction, z3, y3, flattened3, pooled2, unrolled2, y2, flattened2, filter2, pooled1, unrolled1, y1, flattened1 = m.forward_pass(x_test_shrunk)
            print("Loss: ", m.regularized_loss(prediction, y_test_shrunk))
            predicted_classes = np.argmax(prediction, axis=1)
            true_classes = np.argmax(y_test_shrunk, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
            print("Accuracy: ", accuracy)
            print("")
        