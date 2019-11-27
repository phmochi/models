import numpy as np
import scipy.linalg as linalg
from scipy.optimize import basinhopping

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import keras.backend as K
import tensorflow as tf

class ESNRegressor():
    def __init__(self, res_size=1000, leak_rate=0.3, seed=None, burnin=100):
        self.leak_rate = leak_rate
        self.res_size = res_size
        self.burnin = burnin
        if seed:
            self.seed = seed
        
    def init_weights(self, in_size):
        self.w_in = np.random.rand(self.res_size, 1+in_size) - 0.5
        self.w_h = np.random.rand(self.res_size,self.res_size) - 0.5
        
        rho_w = np.max(np.abs(linalg.eig(self.w_h)[0]))
        self.w_h *= 1.25 / rho_w
        
    def solve(self, X, Y, lambd):
        def optfun(beta):
            b = beta.reshape(beta_shape)
#            print(b)
            yhat = np.dot(X_tmp.T, b)
#            print(np.isnan(yhat).sum())
            mse = np.nansum(np.square(Y-yhat) + lambd*np.square(linalg.norm(b)))
            print(mse)
            return mse
            
        beta = np.random.normal(0, 2/(X.shape[0] + Y.shape[1]),
                                (X.shape[0],Y.shape[1])) # xavier initialization      
        
#        beta = np.random.normal(0, 2.5,(X.shape[0],Y.shape[1]))
        
        beta_shape = beta.shape
        beta = beta.ravel()
        
        X_tmp = X.copy()
        X_tmp[np.isnan(X_tmp)] = 0 # nan values do not contribute
        
#        Y[np.isnan(Y)] = 0

#        return minimize(optfun, beta)
        return basinhopping(optfun, beta, disp=1)
        
    def solve2(self, X, Y, lambd, num_epoch=10):
        def rmse(y, yhat):
            nan_mask = tf.logical_not(tf.is_nan(y))
            y = tf.boolean_mask(y, nan_mask)
            yhat = tf.boolean_mask(yhat, nan_mask)
            return K.sqrt(K.mean(K.square((y-yhat))))
            
        X_tmp = X.copy()
        X_tmp[np.isnan(X_tmp)] = 0
        
        Y_tmp = Y.copy()
        Y_tmp[np.isnan(Y_tmp)] = -1
        
        solver = Sequential()
        solver.add(Dense(Y.shape[1], input_dim=X.shape[0], activation="linear", kernel_regularizer=
                         regularizers.l2(lambd)))
        solver.compile(loss=rmse, optimizer="adam")
        
        epoch_scores = [[],[]]
        for _ in range(num_epoch):
#            print(X_tmp.T.shape)
#            print(Y.shape)
            solver.fit(X_tmp.T, Y, epochs=1)
            train_score = np.sqrt(np.nanmean(np.square(solver.predict(X_tmp.T) - Y)))
            
            val_score = np.nan
            if self.Y_valid is not None:
                self.w_out = solver.get_weights()
                predict = self.forecast(num_forecast=self.Y_valid.shape[0])
                val_score = np.sqrt(np.nanmean(np.square(predict - self.Y_valid)))
            
            epoch_scores[0].append(train_score)
            epoch_scores[1].append(val_score)
            print("train_score:",train_score,"validation score:", val_score)
        self.epoch_scores = epoch_scores            
            
        return solver.get_weights()
        
    def fit(self, data, Y, data_valid=None, Y_valid=None, lambd=0.0, mode="nn", 
            num_epoch=10):
        data[np.isnan(data)] = 0
        self.X_valid = data_valid
        self.Y_valid = Y_valid
        
        in_size = data.shape[1]
        sample_size = data.shape[0]
        if self.seed:
            np.random.seed(self.seed)
        self.init_weights(in_size)
        
#        X = np.zeros((in_size+self.res_size+1, sample_size-self.burnin)) 
        X = np.zeros((in_size+self.res_size+1, sample_size)) 
        
        x = np.zeros((self.res_size, 1))
        for t in range(sample_size):
            u = data[t,:].reshape(1,-1)
            
            x = (1-self.leak_rate)*x + self.leak_rate*np.tanh(np.dot(self.w_in,
                np.append(1,u)).reshape(-1,1) + np.dot(self.w_h, x))
            if t >= self.burnin:
                X[:,t-self.burnin] = np.vstack((1,u.T,x))[:,0]

        self.x = x
        self.X = X
        self.u = u
        if mode is "min":
            self.w_out = self.solve(X, Y, lambd)
        else:
            self.w_out = self.solve2(X, Y, lambd, num_epoch=num_epoch)
            
    def forecast(self, num_forecast=14):
        x = self.x.copy()
        u = self.u.copy()
        predictions = np.empty((num_forecast,u.shape[1]))
        for t in range(num_forecast):
#            x = (1-self.leak_rate)*x + self.leak_rate*np.tanh(nandot(self.w_in,
#                np.append(1,u)) + np.dot(self.w_h, x))
            x = (1-self.leak_rate)*x + self.leak_rate*np.tanh(np.dot(self.w_in,
                np.append(1,u)).reshape(-1,1) + np.dot(self.w_h, x))
            
            y = np.dot(self.w_out[0].T, np.vstack((1,u.T,x))).T
            predictions[t,:] = y
            u = y
        return predictions