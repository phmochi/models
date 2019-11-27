from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#from IPython import display
import time
import tensorflow as tf

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
    keep : % to sample into training set
    
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

class Model():
    '''simple model implementation in TF'''
    def __init__(self, input_size, random_state=None):
        self.input_size = input_size
        self.layers = []
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.input = tf.placeholder(tf.float32, (None, input_size))
        self.random_state = random_state
        
    def add_layer(self, size, activation="linear"):
        if self.layers:
            x = self.layers[-1]
        else:
            x = self.input
        W = tf.Variable(self.initializer((x.get_shape().as_list()[1], size)))
        b = tf.Variable(self.initializer((1,size)))
        z = tf.matmul(x,W) + b
        
        if activation == "linear":
            a = z
        elif activation == "relu":
            a = tf.nn.relu(z)
        elif activation == "sigmoid":
            a = tf.nn.sigmoid(z)
        elif activation == "softmax":
            a = tf.nn.softmax(z)
        else:
            raise ValueError("Invalid activation.")
            
        self.layers.append(a)
        
    def compile(self):
        '''currently only supports cross entropy using default adam'''
        yhat = self.layers[-1]
        self.y = tf.placeholder(tf.float32, (None,yhat.get_shape().as_list()[1]))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                    logits=yhat))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
        correct_predictions = tf.equal(tf.argmax(self.y,1), tf.argmax(yhat,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
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
                _, loss = self.sess.run([self.train_step, self.loss], 
                                        feed_dict={self.input: xs, self.y: ys})
                batch_loss.append(loss)
                
            train_loss.append(np.mean(batch_loss))
            
            if validate:
                val_loss.append(self.sess.run(self.loss, feed_dict={self.input: X_val, 
                                                                    self.y: y_val}))
    
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
        return self.sess.run(self.accuracy, feed_dict={self.input: X,
                                                       self.y: y})
        
if __name__ == "__main__":    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train, x_test = preprocess(x_train), preprocess(x_test)
    x_train, x_test = flatten(x_train), flatten(x_test)
    y_train, y_test = encode(y_train), encode(y_test)
    
    x_train, x_valid = split_train_valid(x_train, train_size=5/6, random_state=492)
    y_train, y_valid = split_train_valid(y_train, train_size=5/6, random_state=492)
    
    model = Model(x_train.shape[1], random_state=29)
    model.add_layer(512, "relu")
    model.add_layer(10, "softmax")
    model.compile()
    
    start_time = time.time()
    
    model.fit(x_train, y_train, num_epochs=20, validate=(x_valid, y_valid), 
              early_stop=True)
    
    elapsed_time = time.time() - start_time
    print("time elapsed: %.2f seconds" % elapsed_time)
    print("test evaluation: %.3f" % model.evaluate(x_test, y_test))