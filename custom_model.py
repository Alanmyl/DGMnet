# -*- coding: utf-8 -*-
# file to custom layers, loss and metrics in fitting PDE

from base_config import Config
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.sparse import diags
import numpy as np
import scipy.stats as stats
import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.random.set_seed(1234)
tf.autograph.set_verbosity(1)
class DGM_layer(tf.keras.layers.Layer):

    def __init__(self, units=Config.unit_num):
        super(DGM_layer, self).__init__(trainable=True, name='DGM', dtype=tf.float32)
        self.units = units
        self.matrix = tf.linalg.cholesky(Config.delta * (tf.ones(shape=(Config.dim, Config.dim)) * Config.rho+ tf.eye(Config.dim) * (1-Config.rho)))
    
    def build(self, input_shape):
        ops_size = input_shape[-1]
        self.w1 = self.add_weight(shape=(ops_size, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='w1')
        self.b1 = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True, name='b1')
        self.uzl = self.add_weight(shape=(ops_size, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='uzl')
        self.wzl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='wzl')
        self.bzl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True, name='bzl')
        self.ugl = self.add_weight(shape=(ops_size, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='ugl')
        self.wgl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='wgl')
        self.bgl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True, name='bgl')
        self.url = self.add_weight(shape=(ops_size, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='url')
        self.wrl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='wrl')
        self.brl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True, name='brl')
        self.uhl = self.add_weight(shape=(ops_size, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='uhl')
        self.whl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True, name='whl')
        self.bhl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True, name='bhl')
        self.w = self.add_weight(shape=(self.units, 1), initializer=tf.random_uniform_initializer, trainable=True, name='w')
        self.b = self.add_weight(shape=(1,), initializer=tf.random_uniform_initializer, trainable=True, name='b') 
    
    def act_func(self, x, name=Config.func_name):
        names = ['tanh', 'relu', 'sigmoid']
        assert name in names, "Undefined activation function"
        if name == 'tanh':
            return tf.keras.activations.tanh(x)
        elif name == 'relu':
            return tf.keras.activations.relu(x)
        else:
            return tf.keras.activations.sigmoid(x)

    def sigma(self, x):
        return Config._sig * x
    
    def mu(self, x):
        return Config._mu * x

    def value(self, inputs):
        s = self.act_func(tf.matmul(inputs, self.w1) + self.b1)
        for i in range(Config.layer_num):
            z = self.act_func(tf.matmul(inputs, self.uzl) + tf.matmul(s, self.wzl)  + self.bzl)
            g = self.act_func(tf.matmul(inputs, self.ugl)  + tf.matmul(s, self.wgl) + self.bgl)
            r = self.act_func(tf.matmul(inputs, self.url) + tf.matmul(s, self.wrl) + self.brl)
            h = self.act_func(tf.matmul(inputs, self.uhl)  + tf.matmul(tf.math.multiply(s,r), self.whl) + self.bhl)
            s = tf.math.multiply(1.0 - g, h) + tf.math.multiply(z, s)
        return tf.matmul(s, self.w) + self.b
    
    def boundary(self, x):
        return tf.nn.relu(tf.math.pow(tf.math.reduce_prod(x, axis=1), 1 / tf.cast(tf.shape(x)[-1], tf.float32)))

    def call(self, inputs):
        input1, input2 = tf.split(inputs,2, 0)
        with tf.GradientTape() as g1:
            g1.watch(input1)
            ftx = self.value(input1)
        fprime = g1.gradient(ftx, input1)
        term1_1 = tf.reduce_sum(tf.multiply(tf.concat([self.mu(input1[:,:-1]), tf.ones(shape=(tf.shape(input1)[0], 1))], axis=1), fprime), axis=1)
        xdiff = tfp.distributions.MultivariateNormalTriL(loc=input1[:,:-1], scale_tril=self.matrix).sample(sample_shape=(Config.mc_num))*tf.broadcast_to(self.sigma(input1[:,:-1]),shape=(Config.mc_num,input1.shape[0], input1.shape[1]-1))
        violation = tf.concat([xdiff, tf.zeros(shape=(Config.mc_num, input1.shape[0], 1))], axis=2)
        xplus = tf.expand_dims(input1,axis=0) + violation
        with tf.GradientTape() as g1_1:
            g1_1.watch(xplus)
            ftx_plus = self.value(xplus)
        fprime_plus = g1_1.gradient(ftx_plus, xplus)
        term1_2 = tf.reduce_sum(tf.reduce_mean(tf.multiply((fprime_plus - tf.expand_dims(fprime, axis=0)) / Config.delta, violation), axis=0), axis=1)
        term2 = tf.squeeze(self.value(input2)) - self.boundary(input2[:,:-1])
        #term3 = tf.nn.relu(self.boundary(input3[:,:-1]) - tf.squeeze(self.value(input3)))
        #test = tf.concat([term1_1 + term1_2 * 0.5- Config._r * tf.squeeze(self.value(input1)), term2, term3], axis=0)
        test = tf.concat([term1_1 + term1_2 * 0.5- Config._r * tf.squeeze(self.value(input1)), term2], axis=0)
        return test


train = tf.keras.Input(shape=(Config.dim + 1,), batch_size=Config.batch_size*2)
model = tf.keras.Model(inputs=train, outputs=DGM_layer()(train))
#opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3,decay_steps=2000,decay_rate=0.5,staircase=True))
opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.PiecewiseConstantDecay([250],[1e-3,2e-4]))

class weighted_loss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        loss1, loss2 = tf.split(tf.math.square(y_pred - y_true), 2, axis=0)
        base = tf.maximum(tf.reduce_sum(loss1), tf.reduce_sum(loss2))
        loss1 = loss1 / tf.reduce_sum(loss1) * base
        loss2 = loss2 / tf.reduce_sum(loss2) * base
        return tf.reduce_mean(loss1)/2.0 + tf.reduce_mean(loss2)/2.0

def generate_data(tfixed=False):
    part1 = tf.math.exp(tf.random.uniform(shape=(Config.batch_size, Config.dim), minval=[-Config.xmax for i in range(Config.dim)], maxval=[Config.xmax for i in range(Config.dim)]))
    if tfixed == True:
        part2 = tf.ones(shape=(Config.batch_size, 1)) * Config.T
    else:
        part2 = tf.random.uniform(shape=(Config.batch_size, 1), maxval=[Config.T])
    return tf.concat([part1, part2], axis=1)

import datetime as dt
loss_hist = []
start = dt.datetime.now()
gen_time = dt.datetime.now()-dt.datetime.now()
for i in range(Config.epoch_num):
    mid = dt.datetime.now()
    data1 = generate_data()
    data2 = generate_data()
    #data3 = generate_data(True)
    data = tf.concat([data1, data2], axis=0)
    
    with tf.GradientTape() as tape:
        loss = tf.keras.losses.MeanSquaredError()(0.0, model(data))
        #loss = weighted_loss()(0.0, model(data))
    gen_time = gen_time+dt.datetime.now()-mid
    grads = tape.gradient(loss, model.trainable_weights)
    loss_hist.append(loss.numpy())
    opt.apply_gradients(zip(grads, model.trainable_weights))
    if i%10 == 0:
        print('{} epochs spend {} with loss {:.5f}'.format(i, dt.datetime.now() - start, loss))
        #print('generate {} epochs spend {}'.format(i, gen_time))
    if i == 600:
        model.save_weights(filepath=Config.filepath + "log/model" + str(i + 1))
        break
    

def bs_option(opt_type, r, sig, T, K, s0):
    '''function to calculate options price with Black-Scholes formula.

    Args:
        opt_type: `call` or `put`.
        r: risk-free rate under the risk-neutral measure.
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

sample = tf.concat([generate_data(),generate_data()],axis=0)
option_sig = Config._sig * np.sqrt((1 + (Config.dim - 1) * Config.rho) / Config.dim)
option_mu = Config._mu-Config._sig ** 2/2+option_sig**2/2
eu_opt = bs_option('put', option_mu, option_sig, 0.01, Config.K, np.product(sample[:-1],axis=1))
res = model.get_layer(name='DGM').value(sample[:Config.batch_size])
print(model.predict(sample))
import matplotlib.pyplot as plt
plt.plot(loss_hist)
plt.savefig(Config.filepath+'loss.png')
print(loss_hist)
import pandas as pd
pd.Series(loss_hist).to_csv(Config.filepath+'loss.csv')

'''
# finite difference method for pricing American options
N, Nt = int(1e5), int(2e3)
h, tau = Config.xmax / N, Config.T / Nt
gamma1 = (Config._mu - Config._sig ** 2 / 2) / 2 * tau / h  - Config._sig ** 2 * tau / h ** 2 / 2
gamma2 = 1 - Config._sig ** 2 * tau / h ** 2
gamma3 = -(Config._mu - Config._sig ** 2 / 2) / 2 * tau / h  - Config._sig ** 2 * tau / h ** 2 / 2
C = diags([[gamma1],[gamma2],[gamma3]], [-1, 0, 1], shape=(N+1, N+1))
'''