from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import time

def preprocess(X):
    '''scales pixels to be between 0 and 1'''
    return X / 255.

def encode(v):
    '''onehot encode a vector'''
    possible_values = np.unique(v)
    encoded = np.zeros((v.shape[0], len(possible_values)))
    for i, val in enumerate(possible_values):
        encoded[v==val,i] = 1
    return encoded

def split_train_valid(X, train_size, random_state=None):
    ''' Splits a sample into a training and validation set 
    
    Parameters
    ----------
    train_size : % to sample into training set
    
    '''
    num_sample = int(train_size*X.shape[0])
    if random_state:
        np.random.seed(random_state)
        
    idx = np.random.choice(X.shape[0], replace=False, size=num_sample)
    mask = np.zeros(X.shape[0], dtype="bool")
    mask[idx] = 1
    
    return X[mask,:], X[~mask,:]

def flatten(X):
    '''flatten n-d array to 1-d vector'''
    dim = np.multiply.reduce(X.shape[1:])
    return np.ravel(X).reshape(-1,dim)

class Model:
    '''simple model implementation in numpy
    
    Last layer corresponds to softmax: Don't calculate it automatically on
    forward pass as we want to calculate the loss directly without calculating
    the probabilities for better numerical stability.
    
    Implements:
            1. ReLU
            2. L2 Regularization
            3. Stochastic Gradient Descent
            4. Early Stopping
            5. Random/Xavier/He Initialization
            
    TODO:
            1. Implement optimizer class
    '''
    def __init__(self, random_state=None):
        self.layers = []
        self.random_state = random_state
        
    class Layer(metaclass=ABCMeta):
        @abstractmethod
        def __init__(self):
            pass
        @abstractmethod
        def forward(self):
            pass
        @abstractmethod
        def backward(self, inputs, grads):
            pass
        
    class Dense(Layer):
        def __init__(self, in_size, out_size, alpha, init="he", lr=1e-1):
            if init == "random":
                self.W = np.random.randn(in_size,out_size) * .01
            elif init == "xavier":
                self.W = np.random.randn(in_size,out_size) * 2/(in_size+out_size)
            elif init == "he":
                self.W = np.random.randn(in_size,out_size) * 2/in_size
            else:
                raise ValueError("Invalid initializer.")

            self.b = np.zeros(out_size)
            self.lr = lr
            self.alpha = alpha
            
        def forward(self, X):
            return X.dot(self.W) + self.b
        
        def backward(self, inputs, grads):
            grad_input = grads.dot(self.W.T)
            
            grad_weights = inputs.T.dot(grads)
            grad_biases = np.sum(grads.dot(np.eye(self.b.shape[0])), axis=0)
            
            self.W = self.W - self.lr * grad_weights
            self.b = self.b - self.lr * grad_biases
            return grad_input          
        
        def l2_reg(self):
            return self.alpha*np.sum(self.W**2)
        
    class ReLU(Layer):
        def __init__(self):
            pass
        
        def forward(self, X):
            return np.maximum(X,0)
        
        def backward(self, inputs, grads):
            return grads * (inputs > 0)
        
    def calc_crossentropy(self, labels, logits):
        ce =  -(labels*logits).sum(axis=1,keepdims=True) + \
            np.log(np.exp(logits).sum(axis=1,keepdims=True))
            
        reg_penalty = sum(x.l2_reg() for x in self.layers if isinstance(x,self.Dense))
        ce += reg_penalty
        return ce
        
    def calc_crossentropy_grad(self, labels, logits):
        return (-labels + np.exp(logits)/np.exp(logits).sum(
                axis=1, keepdims=True)) / labels.shape[0]
        
    def add_dense(self, in_size, out_size, alpha=0.0):
        self.layers.append(self.Dense(in_size, out_size, alpha))
        
    def forward_pass(self, X):
        self.inputs = []
        x = X
        self.inputs.append(x)
        for l in self.layers:
            x = l.forward(x)
            self.inputs.append(x)
        return x
    
    def backward_pass(self, grads):
        cost_grad = grads
        for inputs, l in zip(self.inputs[:-1][::-1],self.layers[::-1]):
            cost_grad = l.backward(inputs, cost_grad)
        
    def add_activation(self, how):
        if how == "relu":
            self.layers.append(self.ReLU())
        else:
            raise ValueError("Invalid activation.")
        
    def get_batches(self, X, y, n):        
        if self.random_state:
            np.random.seed(self.random_state)
        l = len(X)
        idx = np.arange(l)
        np.random.shuffle(idx)
        X = X[idx,:]
        y = y[idx,:]
        for ndx in range(0, l, n):
            yield (X[ndx:min(ndx+n, l)], y[ndx:min(ndx+n, l)])
    
    def plot_scores(self, train, val, title=None):
        plt.figure(figsize=(13,9))
        plt.plot(np.arange(1,len(train)+1), train, label="Train")
        if val:
            plt.plot(np.arange(1,len(train)+1), val, label="Validation")
        if title:
            plt.title(title)
        plt.legend()
        plt.show()
#        display.clear_output(wait=True) #for jupyter notebook
    
    def fit(self, X, y, num_epochs=1, early_stop=False, patience=10, 
            validate=None, batch_size=32):
        train_loss = []
        val_loss = []
        wait = 0
        best_loss = 9999
        if validate:
            X_val, y_val = validate
            
        for i in range(1, num_epochs+1):
            batch_loss = []
            for xs, ys in self.get_batches(X, y, batch_size):
                logits = self.forward_pass(xs)
                loss = np.mean(self.calc_crossentropy(ys, logits))
                grads = self.calc_crossentropy_grad(ys, logits)

                self.backward_pass(grads)
                batch_loss.append(loss)
                
            train_loss.append(np.mean(batch_loss))
            
            if validate:
                val_score = np.mean(self.calc_crossentropy(y_val, 
                                                           self.forward_pass(X_val)))
                val_loss.append(val_score)
    
            self.plot_scores(train_loss, val_loss, title="Loss")
            
            if validate and early_stop:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    wait = 0
                else:
                    wait += 1
                
                if wait > patience:
                    break
                
    def evaluate(self, X, y):
        '''calculates accuracy'''
        return np.mean(self.predict(X) == np.argmax(y, axis=1))
    
    def predict(self, X):
        return np.argmax(self.forward_pass(X), axis=1)

if __name__ == "__main__":        
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train, x_test = preprocess(x_train), preprocess(x_test)
    x_train, x_test = flatten(x_train), flatten(x_test)
    y_train, y_test = encode(y_train), encode(y_test)
    
    x_train, x_valid = split_train_valid(x_train, train_size=5/6, random_state=492)
    y_train, y_valid = split_train_valid(y_train, train_size=5/6, random_state=492)
    
    model = Model(random_state=29)
    model.add_dense(x_train.shape[1], 512)
    model.add_activation("relu")
    model.add_dense(512, 10)
    
    start_time = time.time()
    
    model.fit(x_train, y_train, num_epochs=5, validate=(x_valid, y_valid), 
              early_stop=True)
    
    elapsed_time = time.time() - start_time
    print("time elapsed: %.2f seconds" % elapsed_time)
    print("test evaluation: %.3f" % model.evaluate(x_test, y_test))