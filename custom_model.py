 
# -*- coding: utf-8 -*-
# file to custom layers, loss and metrics in fitting PDE

from base_config import Config
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.sparse import diags
import numpy as np
import scipy.stats as stats
import datetime as dt

# avoid unnecessary warning from tensorflow
import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.random.set_seed(143)
tf.autograph.set_verbosity(1)

# define the the layer in DGM NN
class DGM_layer(tf.keras.layers.Layer):
    """The only layer to process input in the NN.
    Args:
        units: number of neurons of NN
    Attributes:
        units: number of neurons of NN
        matrix: calculate cholesky decomposition of covariance matrix of normal distribution repeatedly used in Monte Carlo
    """
    def __init__(self, units=Config.unit_num):
        super(DGM_layer, self).__init__(trainable=True, name='DGM', dtype=tf.float32)
        self.units = units
        # avoid unnecessary calculation for high-dimensional x, because they share the same matrix given the length of x
        self.matrix = tf.linalg.cholesky(Config.delta * (tf.ones(shape=(Config.dim, Config.dim)) * Config.rho+ tf.eye(Config.dim) * (1-Config.rho)))
    
    def build(self, input_shape):
        """A build-in function of tensorflow to initialize the weights in NN.
        Args:
            input_shape: default parameter 'input_shape'of class `layer`
        """
        ops_size = input_shape[-1]
        init = tf.keras.initializers.GlorotNormal # set initializer function for parameters
        self.w1 = self.add_weight(shape=(ops_size, self.units), initializer=init, trainable=True, name='w1')
        self.b1 = self.add_weight(shape=(self.units,), initializer=init, trainable=True, name='b1')
        self.uz, self.wz, self.bz = [], [], []
        self.ug, self.wg, self.bg = [], [], []
        self.ur, self.wr, self.br = [], [], []
        self.uh, self.wh, self.bh = [], [], []
        for i in range(Config.layer_num):
            self.uz.append(self.add_weight(shape=(ops_size, self.units), initializer=init, trainable=True, name='uz{}'.format(i)))
            self.wz.append(self.add_weight(shape=(self.units, self.units), initializer=init, trainable=True, name='wz{}'.format(i)))
            self.bz.append(self.add_weight(shape=(self.units,), initializer=init, trainable=True, name='bz{}'.format(i)))
            self.ug.append(self.add_weight(shape=(ops_size, self.units), initializer=init, trainable=True, name='ug{}'.format(i)))
            self.wg.append(self.add_weight(shape=(self.units, self.units), initializer=init, trainable=True, name='wg{}'.format(i)))
            self.bg.append(self.add_weight(shape=(self.units,), initializer=init, trainable=True, name='bg{}'.format(i)))
            self.ur.append(self.add_weight(shape=(ops_size, self.units), initializer=init, trainable=True, name='ur{}'.format(i)))
            self.wr.append(self.add_weight(shape=(self.units, self.units), initializer=init, trainable=True, name='wr{}'.format(i)))
            self.br.append(self.add_weight(shape=(self.units,), initializer=init, trainable=True, name='br{}'.format(i)))
            self.uh.append(self.add_weight(shape=(ops_size, self.units), initializer=init, trainable=True, name='uh{}'.format(i)))
            self.wh.append(self.add_weight(shape=(self.units, self.units), initializer=init, trainable=True, name='wh{}'.format(i)))
            self.bh.append(self.add_weight(shape=(self.units,), initializer=init, trainable=True, name='bh{}'.format(i)))
        self.w = self.add_weight(shape=(self.units, 1), initializer=init, trainable=True, name='w')
        self.b = self.add_weight(shape=(1,), initializer=init, trainable=True, name='b') 
    
    def act_func(self, x, name=Config.func_name):
        """select the type of activation function in NN.
        Args
            x: value of input
            name: type of activation function
        """
        names = ['tanh', 'relu', 'sigmoid']
        assert name in names, "Undefined activation function"
        if name == 'tanh':
            return tf.keras.activations.tanh(x)
        elif name == 'relu':
            return tf.keras.activations.relu(x)
        else:
            return tf.keras.activations.sigmoid(x)

    def sigma(self, x):
        """set function $\sigma(x)$ in PDE
        """
        return Config._sig * x
    
    def mu(self, x):
        """set function $\mu(x)$ in PDE
        """
        return Config._mu * x

    def value(self, inputs):
        """estimated function of $f$; generally, `inputs` here has 1/3 length for American opyion and 1/2 for European option
        Args:
            inputs: array $(x,t)$
        Returns:
            the value of the function $f$
        """
        s = self.act_func(tf.matmul(inputs, self.w1) + self.b1)
        for i in range(Config.layer_num):
            z = self.act_func(tf.matmul(inputs, self.uz[i]) + tf.matmul(s, self.wz[i])  + self.bz[i])
            g = self.act_func(tf.matmul(inputs, self.ug[i])  + tf.matmul(s, self.wg[i]) + self.bg[i])
            r = self.act_func(tf.matmul(inputs, self.ur[i]) + tf.matmul(s, self.wr[i]) + self.br[i])
            h = self.act_func(tf.matmul(inputs, self.uh[i]) + tf.matmul(tf.math.multiply(s, r), self.wh[i]) + self.bh[i])
            s = tf.math.multiply(1.0 - g, h) + tf.math.multiply(z, s)
        return tf.matmul(s, self.w) + self.b
    
    def boundary(self, x):
        """terminal condition of the PDE
        """
        return tf.nn.relu(Config.K-tf.math.pow(tf.math.reduce_prod(x, axis=1), 1.0  / tf.cast(tf.shape(x)[-1], tf.float32)))

    def call(self, inputs):
        """function which will be applied when the class is called.
        Args:
            inputs: concatenated array with data from 2/3 datasets. The first is for dynamic term, second for terminal term, third for constraint term when it's splitted.
        Returns:
            concatenated array including values for every term, the purpose is to make the values to be 0.
        """
        input1, input2 = tf.split(inputs,2, 0)
        with tf.GradientTape() as g1:
            g1.watch(input1)
            ftx = self.value(input1)
        fprime = g1.gradient(ftx, input1)
        # terms containing the first derivatives
        term1_1 = tf.reduce_sum(tf.multiply(tf.concat([self.mu(input1[:,:-1]), tf.ones(shape=(tf.shape(input1)[0], 1))], axis=1), fprime), axis=1)
        xdiff = tfp.distributions.MultivariateNormalTriL(loc=0.0, scale_tril=self.matrix).sample(sample_shape=(Config.mc_num, input1.shape[0])) * tf.broadcast_to(self.sigma(input1[:,:-1]), shape=(Config.mc_num, input1.shape[0], input1.shape[1] - 1))
        violation = tf.concat([xdiff, tf.zeros(shape=(Config.mc_num, input1.shape[0], 1))], axis=2)
        xplus = tf.expand_dims(input1,axis=0) + violation
        with tf.GradientTape() as g1_1:
            g1_1.watch(xplus)
            ftx_plus = self.value(xplus)
        fprime_plus = g1_1.gradient(ftx_plus, xplus)
        # terms containing the second derivatives and r*f
        term1_2 = tf.reduce_sum(tf.reduce_mean(tf.multiply((fprime_plus - tf.expand_dims(fprime, axis=0)) / Config.delta, violation), axis=0), axis=1)
        term2 = tf.squeeze(self.value(input2)) - self.boundary(input2[:,:-1])
        #term3 = tf.nn.relu(self.boundary(input3[:,:-1]) - tf.squeeze(self.value(input3)))
        #test = tf.concat([term1_1 + term1_2 * 0.5- Config._r * tf.squeeze(self.value(input1)), term2, term3], axis=0)
        return tf.concat([term1_1 + term1_2 * 0.5- Config._r * tf.squeeze(self.value(input1)), term2], axis=0)

# build the model
train = tf.keras.Input(shape=(Config.dim + 1,), batch_size=Config.batch_size*2)
model = tf.keras.Model(inputs=train, outputs=DGM_layer()(train))
#opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3,decay_steps=2000,decay_rate=0.5,staircase=True)) tf.keras.optimizers.schedules.PiecewiseConstantDecay([300],[1e-3,2e-4])
#opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.PiecewiseConstantDecay([5000, 10000, 20000, 30000, 40000, 45000],[1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]))
opt = tf.keras.optimizers.Adam()

class weighted_loss(tf.keras.losses.Loss):
    """customed loss function, isn't used
    """
    def call(self, y_true, y_pred):
        loss1, loss2 = tf.split(tf.math.square(y_pred - y_true), 2, axis=0)
        return tf.reduce_mean(loss1)/2.0+tf.reduce_mean(loss2)/2.0

# generate trainset
def generate_data(tfixed:bool):
    """generate required data to train the model
    Args:
        if tfixed is True, the data is for terminal condition with `t` as `T`, otherwise it's for dynamic condition or constraint condition
    Returns:
        generated data
    """
    # sampling x
    part1 = tf.math.exp(tf.random.uniform(shape = (Config.batch_size, Config.dim), minval = [-Config.xmax for i in range(Config.dim)], maxval = [Config.xmax for i in range(Config.dim)]))
    # sampling t
    if tfixed == True:
        part2 = tf.ones(shape=(Config.batch_size, 1)) * Config.T
    else:
        part2 = tf.random.uniform(shape=(Config.batch_size, 1), maxval=[Config.T])
    return tf.concat([part1, part2], axis=1)

# custom training process, which can be integrated in the class definition of model as `train_step` in the coming tensorflow2.2.0
loss_hist = []
start = dt.datetime.now()
gen_time = dt.datetime.now() - dt.datetime.now()
try:
    for i in range(Config.epoch_num):
        mid = dt.datetime.now()
        data1 = generate_data(False)
        data2 = generate_data(True)
        #data3 = generate_data(True)
        data = tf.concat([data1, data2], axis=0)
        with tf.GradientTape() as tape:
            #loss = tf.keras.losses.MeanSquaredError()(0.0, model(data))
            loss = weighted_loss()(0.0, model(data))
        #gen_time = gen_time+dt.datetime.now()-mid
        grads = tape.gradient(loss, model.trainable_weights)
        loss_hist.append(loss.numpy())
        opt.apply_gradients(zip(grads, model.trainable_weights))
        if i%10 == 0:
            print('{} epochs spend {} with loss {:.5f}'.format(i, dt.datetime.now() - start, loss))
            #print('generate {} epochs spend {}'.format(i, gen_time))
        if (i+1)%1000 == 0:
            model.save_weights(filepath=Config.filepath + "log/model" + str(i + 1))
except KeyboardInterrupt:
    pass
finally:
    import matplotlib.pyplot as plt
    plt.plot(loss_hist)
    plt.savefig(Config.filepath+'loss.png')
    import pandas as pd
    pd.Series(loss_hist).to_csv(Config.filepath + 'loss.csv')
    model.save_weights(filepath=Config.filepath + "log/final_model" )

# test the training result with analytical solution
def bs_option(opt_type, r, sig, T, K, s0):
    '''function to calculate options price with Black-Scholes formula.
    Args:
        opt_type: `call` or `put`.
        r: 
        sig: volatility.
        T: time to maturity.
        K: strike price.
        s0:initial price.
    Returns:
        A value or a sequence of values with datetime index.
    '''
    assert opt_type in ['call', 'put'], 'Wrong argument '+opt_type
    d1 = (np.log(s0 / K) + (r + sig ** 2 / 2) * T) / sig / T ** 0.5
    d2 = d1 - sig * T ** 0.5
    if opt_type == 'put':
        return stats.norm.cdf(-d2) * K * np.exp(-r * T) - stats.norm.cdf(-d1) * s0
    else:
        return stats.norm.cdf(d1) * s0 - stats.norm.cdf(d2) * K * np.exp(-r * T)

sample = tf.concat([generate_data(False),generate_data(True)],axis=0)
option_sig = Config._sig * np.sqrt((1 + (Config.dim - 1) * Config.rho) / Config.dim)
option_mu = Config._mu-Config._sig ** 2/2+option_sig**2/2
eu_opt = bs_option('put', option_mu, option_sig, Config.T-sample[:,-1], Config.K, np.power(np.product(sample[:,:-1],axis=1),1/Config.dim))[:20]
res = model.get_layer(name='DGM').value(sample[:20])