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
        # outputs shape of batch_size x 512
        self.batch_size = 32
        self.w1_self = np.random.randn(512, 512) * np.sqrt(2.0 / 1024)
        self.w1_input = np.random.randn(vocab_size, 512) * np.sqrt(2.0 / (512 + vocab_size))
        self.b1 = np.zeros(512)
        self.h1_state = np.zeros((self.batch_size, 512)) #an activation of length 512 per batch
        
        #output batch_size x 288
        self.w2_self = np.random.randn(288, 288) * np.sqrt(2.0 / 576)
        self.w2_input = np.random.randn(512, 288) * np.sqrt(2.0 / 800)
        self.b2 = np.zeros(288)
        self.h2_state = np.zeros((self.batch_size, 288)) #activation of length 288 per batch
        
        #output batch_size x 95
        self.w_out = np.random.randn(288, vocab_size) * np.sqrt(2.0 / (288 + vocab_size))
        self.b_out = np.zeros(vocab_size)
        
        #adam stuff
        self.w1_self_m = np.zeros((512, 512))
        self.w1_self_v = np.zeros((512, 512))
        self.w1_input_m = np.zeros((vocab_size, 512))
        self.w1_input_v = np.zeros((vocab_size, 512))
        self.b1_m = np.zeros(512)
        self.b1_v = np.zeros(512)
        
        self.w2_self_m = np.zeros((288, 288))
        self.w2_self_v = np.zeros((288, 288))
        self.w2_input_m = np.zeros((512, 288))
        self.w2_input_v = np.zeros((512, 288))
        self.b2_m = np.zeros(288)
        self.b2_v = np.zeros(288)
        
        self.w_out_m = np.zeros((288, vocab_size))
        self.w_out_v = np.zeros((288, vocab_size))
        self.b_out_m = np.zeros(vocab_size)
        self.b_out_v = np.zeros(vocab_size)
        
    def softmax(self, y):
        y_shifted = y - np.max(y, axis=-1, keepdims=True)
        exp_y = np.exp(y_shifted)
        return exp_y / np.sum(exp_y, axis=-1, keepdims=True)
    
    def cross_entropy_loss(self, prediction, target):
        return np.mean(np.sum(-target * np.log(prediction + 1e-12), axis = -1))
    
    def regularized_loss(self, prediction, target):
        return (self.cross_entropy_loss(prediction, target) 
                + reg_coef * (np.sum(self.w1_self ** 2) + np.sum(self.w1_input ** 2) + np.sum(self.b1 ** 2)
                              + np.sum(self.w2_self ** 2) + np.sum(self.w2_input ** 2) + np.sum(self.b2 ** 2)
                              + np.sum(self.w_out ** 2) + np.sum(self.b_out ** 2)))

    def forward_pass(self, batch):
        z1 = self.h1_state @ self.w1_self + batch @ self.w1_input+ self.b1 #batch_size x 512
        self.h1_state = np.tanh(z1) #batch_size x 512
        z2 = self.h2_state @ self.w2_self + self.h1_state @ self.w2_input + self.b2 #shape batch_size x 288
        self.h2_state = np.tanh(z2) # batch_size x 288
        z3 = self.h2_state @ self.w_out + self.b_out #y3 has shape batch_size x 95
        prediction = self.softmax(z3)
        return z1, z2, z3, prediction
    
    def backward_pass(self, prediction, target, z2, z1, batch, l2_hidden_states, l1_hidden_states):
        batch_size = self.batch_size
        z3_bar = (prediction - target) / batch_size #batch_size x 95
        b_out_bar = np.sum(z3_bar, axis = (0, 1)) + reg_coef * self.b_out #95x1
        #we are going to do BPTT over say the past 100 characters
        #need to accumulate the gradient
        
        z2_bar = np.zeros((batch_size, 288))
        h2_state_bar = np.zeros((batch_size, 288))
        w2_self_bar = np.zeros((288, 288))
        w2_input_bar = np.zeros((512, 288))
        b2_bar = np.zeros((288))
        
        z1_bar = np.zeros((batch_size, 512))
        h1_state_bar = np.zeros((batch_size, 512))
        w1_self_bar = np.zeros((512, 512))
        w1_input_bar = np.zeros((vocab_size, 512))
        b1_bar = np.zeros((512))
        
        w_out_bar = np.zeros((288, vocab_size))
        for i in reversed(range(100)):
            w_out_bar += l2_hidden_states[i + 1].T @ z3_bar[i]
            l2_i = l2_hidden_states[i] #batch_size x 288
            l1_i = l1_hidden_states[i] #batch_size x 512
            
            h2_state_bar = z3_bar[i] @ self.w_out.T + z2_bar @ self.w2_self.T # batch_size x 288 
            z2_bar = h2_state_bar * (1 - z2[i] ** 2) #batch_sizex288
            z2_bar = np.clip(z2_bar, -1.0, 1.0)
            w2_self_bar += z2_bar.T @ l2_i #288x288
            w2_input_bar += l1_i.T @ z2_bar #512x288
            b2_bar += np.sum(z2_bar, axis=0) #288x1
            
            h1_state_bar = z2_bar @ self.w2_input.T + z1_bar @ self.w1_self.T #batch_size x 512
            z1_bar = h1_state_bar * (1 - z1[i] ** 2) #batch_size x 512
            z1_bar = np.clip(z1_bar, -1.0, 1.0)
            w1_self_bar += z1_bar.T @ l1_i #512x512
            w1_input_bar += np.eye(vocab_size)[inputs[:, i]].T @ z1_bar #95x512
            b1_bar += np.sum(z1_bar, axis=0)

        #average over chunk and regularize
        w2_self_bar = w2_self_bar / chunk_size + self.w2_self * reg_coef
        w2_input_bar = w2_input_bar / chunk_size + self.w2_input * reg_coef
        b2_bar = b2_bar / chunk_size + self.b2 * reg_coef
        w1_self_bar = w1_self_bar / chunk_size + self.w1_self * reg_coef
        w1_input_bar = w1_input_bar / chunk_size + self.w1_input * reg_coef
        b1_bar = b1_bar / chunk_size + self.b1 * reg_coef
        w_out_bar = w_out_bar / chunk_size + self.w_out * reg_coef
        
        return b1_bar, w1_input_bar, w1_self_bar, b2_bar, w2_input_bar, w2_self_bar, w_out_bar, b_out_bar
        
    def clip_gradients(self, grads, max_norm=5.0):
        total_norm = np.sqrt(sum(np.linalg.norm(g.ravel())**2 for g in grads))
        scale = min(1.0, max_norm / (total_norm + 1e-6))
        return [g * scale for g in grads]

        
    def adam(self, prediction, target, z2, z1, batch, l2_hidden_states, l1_hidden_states):
        (b1_bar, w1_input_bar, w1_self_bar, b2_bar, 
         w2_input_bar, w2_self_bar, w_out_bar, b_out_bar) = self.backward_pass(prediction, target, z2, z1, batch, l2_hidden_states, l1_hidden_states)
        grads = [w1_self_bar, w1_input_bar, b1_bar, w2_self_bar, w2_input_bar, b2_bar, w_out_bar, b_out_bar]
        grads = self.clip_gradients(grads)
        w1_self_bar, w1_input_bar, b1_bar, w2_self_bar, w2_input_bar, b2_bar, w_out_bar, b_out_bar = grads

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
        
    def sample(self, seed_char, length=500, temperature=1.0):
        h1_saved, h2_saved = self.h1_state.copy(), self.h2_state.copy()
        bs = self.batch_size
        self.batch_size = 1
        self.h1_state = np.zeros((1, 512))
        self.h2_state = np.zeros((1, 288))

        result = [seed_char]
        x = np.eye(vocab_size)[char_to_idx[seed_char]].reshape(1, -1)

        for _ in range(length):
            _, _, _, pred = self.forward_pass(x)
            logits = np.log(pred[0] + 1e-12) / temperature
            probs = np.exp(logits) / np.exp(logits).sum()
            idx = np.random.choice(vocab_size, p=probs)
            result.append(idx_to_char[idx])
            x = np.eye(vocab_size)[idx].reshape(1, -1)

        # restore training hidden states
        self.h1_state, self.h2_state = h1_saved, h2_saved
        self.batch_size = bs
        return ''.join(result)

reg_coef = 1e-4
beta = 0.9
gamma = 0.999
alpha = 0.001

with open('RNN/earth_science.txt', 'r') as f:
    text = f.read()

import string
text = ''.join(c for c in text if c in string.printable)

# vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# encode entire text as integers
data = np.array([char_to_idx[c] for c in text])

# one hot
def one_hot(idx, vocab_size):
    x = np.zeros(vocab_size)
    x[idx] = 1.0
    return x

rnn = model()
rnn.batch_size = 32
chunk_size = 100

trim = len(data) - (len(data) % rnn.batch_size)
data = data[:trim]
streams = data.reshape(rnn.batch_size, -1)

# now chunk each stream
for j in range(250): #250 epochs
    for i in range(0, streams.shape[1] - chunk_size, chunk_size):
        inputs  = streams[:, i:i+chunk_size] # (batch_size, chunk_size)
        targets = streams[:, i+1:i+chunk_size+1] # shifted by 1
        targets_onehot = np.eye(vocab_size)[targets]  # (batch_size, chunk_size, vocab_size)
        targets_onehot = targets_onehot.transpose(1, 0, 2)  # (chunk_size, batch_size, vocab_size)        
        hidden_states_l1 = np.zeros((chunk_size + 1, rnn.batch_size, 512))
        hidden_states_l1[0] = rnn.h1_state
        hidden_states_l2 = np.zeros((chunk_size + 1, rnn.batch_size, 288))
        hidden_states_l2[0] = rnn.h2_state
        predictions = np.zeros((chunk_size, rnn.batch_size, vocab_size))
        z1s = np.zeros((chunk_size, rnn.batch_size, 512))
        z2s = np.zeros((chunk_size, rnn.batch_size, 288))
        for t in range(chunk_size):
            x = inputs[:, t]
            # onehot: (batch_size, vocab_size)
            x_onehot = np.eye(vocab_size)[x]
            z1, z2, z3, prediction = rnn.forward_pass(x_onehot)
            hidden_states_l1[t + 1] = rnn.h1_state
            hidden_states_l2[t + 1] = rnn.h2_state
            z1s[t] = z1
            z2s[t] = z2
            predictions[t] = prediction
        rnn.adam(predictions, targets_onehot, z2s, z1s, inputs, hidden_states_l2, hidden_states_l1)
        if (i % 3400 == 0):
            print("Loss = ", rnn.regularized_loss(predictions, targets_onehot))
    if (j % 10 == 0):
        print("Epoch ", j)
        print(rnn.sample('T'))
