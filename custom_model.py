# -*- coding: utf-8 -*-
# file to custom layers, loss and metrics in fitting PDE

from base_config import Config
import tensorflow as tf
import tensorflow_probability as tfp

class lstm_layer(tf.keras.layers.Layer):

    def __init__(self, units=Config.unit_num):
        super(lstm_layer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.b1 = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True)
        self.uzl = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.wzl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.bzl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True)
        self.ugl = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.wgl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.bgl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True)
        self.url = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.wrl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.brl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True)
        self.uhl = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.whl = self.add_weight(shape=(self.units, self.units), initializer=tf.random_uniform_initializer, trainable=True)
        self.bhl = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer, trainable=True)
        self.w = self.add_weight(shape=(self.units, 1), initializer=tf.random_uniform_initializer, trainable=True)
        self.b = self.add_weight(shape=(1,), initializer=tf.random_uniform_initializer, trainable=True)

    def act_func(self, x, name=Config.func_name):
        names = ['tanh', 'relu', 'sigmoid']
        assert name in names, "Undefined activation function"
        return tf.switch_case(tf.constant(names.index(name)), branch_fns={0: lambda: tf.keras.activations.tanh(x), 1: lambda: tf.keras.activations.relu(x), 2: lambda: tf.keras.activations.sigmoid(x)})

    def call(self, inputs):
        s = self.act_func(tf.matmul(inputs, self.w1) + self.b1)
        for i in range(Config.layer_num):
            z = self.act_func(tf.matmul(inputs, self.uzl) + tf.matmul(s, self.wzl)  + self.bzl)
            g = self.act_func(tf.matmul(inputs, self.ugl)  + tf.matmul(s, self.wgl) + self.bgl)
            r = self.act_func(tf.matmul(inputs, self.url) + tf.matmul(s, self.wrl) + self.brl)
            h = self.act_func(tf.matmul(inputs, self.uhl)  + tf.matmul(tf.math.multiply(s,r), self.whl) + self.bhl)
            s = tf.math.multiply(1.0 - g, h) + tf.math.multiply(z, s)
        return tf.matmul(s, self.w) + self.b
    '''
    def dynamic_pred(self, inputs):
        print(inputs)
        with tf.GradientTape() as g1:
            g1.watch(inputs)
            ftx = self.value(inputs)
        fprime = g1.gradient(ftx, inputs)
        term1_1 = tf.reduce_sum(tf.multiply(tf.concat([self.mu(inputs[:,:-1]), tf.ones(shape=(tf.shape(inputs)[0], 1))], axis=1), fprime), axis=1)
        matrix = Config.delta * (tf.ones(shape=(inputs.shape[1] - 1, inputs.shape[1] - 1)) * 0.75+ tf.eye(inputs.shape[1] - 1) * 0.25)
        cov = lambda x: tfp.distributions.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(matrix)).sample(sample_shape=(Config.mc_num))*tf.broadcast_to(self.sigma(x),shape=(Config.mc_num, inputs.shape[-1]-1))
        violation = tf.concat([tf.vectorized_map(cov, inputs[:,:-1]), tf.zeros(shape=(inputs.shape[0], Config.mc_num,1))],axis=2)
        xplus = tf.expand_dims(inputs,axis=1) + violation
        with tf.GradientTape() as g1_1:
            g1_1.watch(xplus)
            ftx_plus = self.value(xplus)
        fprime_plus = g1_1.gradient(ftx_plus, xplus)
        term1_2 = tf.reduce_sum(tf.reduce_mean(tf.multiply((fprime_plus - tf.expand_dims(fprime, axis=1)) / Config.delta, violation), axis=2), axis=1)
        return term1_1 + term1_2 * 0.5- Config._r * tf.squeeze(self.value(inputs))
    
    def constraint_pred(self, inputs):
        return tf.nn.relu(self.boundary(inputs[:,:-1]) - tf.squeeze(self.value(inputs)))

    def boundary_pred(self, inputs):
        return tf.squeeze(self.value(inputs))-self.boundary(inputs[:,:-1])
    
    def call(self, inputs):
        res = []
        for i in range(3):
            print(tf.equal(tf.cast(inputs[:, -1], tf.int32), i))
            print(inputs)
            data = tf.boolean_mask(inputs, tf.equal(tf.cast(inputs[:, -1], tf.int32), tf.constant(i)))
            print(data)
            res.append(tf.switch_case(tf.cast(data[0,-1], tf.int32), branch_fns={0: lambda: self.dynamic_pred(data[:, :-1]), 1: lambda: self.constraint_pred(data[:, :-1]), 2: lambda: self.boundary_pred(data[:, :-1])}))
        return tf.concat(res,axis=0)'''
        
class dynamic_layer(tf.keras.layers.Layer):

    def __init__(self, units=Config.unit_num):
        super(dynamic_layer, self).__init__()
        self.units = units

    def sigma(self, x):
        return Config._sig * x
    
    def mu(self, x):
        return Config._mu * x

    def call(self, inputs, layer):
        with tf.GradientTape() as g1:
            g1.watch(inputs)
            ftx = layer(inputs)
        fprime = g1.gradient(ftx, inputs)
        term1_1 = tf.reduce_sum(tf.multiply(tf.concat([self.mu(inputs[:,:-1]), tf.ones(shape=(tf.shape(inputs)[0], 1))], axis=1), fprime), axis=1)
        matrix = Config.delta * (tf.ones(shape=(inputs.shape[1] - 1, inputs.shape[1] - 1)) * 0.75+ tf.eye(inputs.shape[1] - 1) * 0.25)
        cov = lambda x: tfp.distributions.MultivariateNormalTriL(loc=x, scale_tril=tf.linalg.cholesky(matrix)).sample(sample_shape=(Config.mc_num))*tf.broadcast_to(self.sigma(x),shape=(Config.mc_num, inputs.shape[-1]-1))
        violation = tf.concat([tf.vectorized_map(cov, inputs[:,:-1]), tf.zeros(shape=(inputs.shape[0], Config.mc_num,1))],axis=2)
        xplus = tf.expand_dims(inputs,axis=1) + violation
        with tf.GradientTape() as g1_1:
            g1_1.watch(xplus)
            ftx_plus = layer(xplus)
        fprime_plus = g1_1.gradient(ftx_plus, xplus)
        term1_2 = tf.reduce_sum(tf.reduce_mean(tf.multiply((fprime_plus - tf.expand_dims(fprime, axis=1)) / Config.delta, violation), axis=2), axis=1)
        return term1_1 + term1_2 * 0.5- Config._r * tf.squeeze(layer(inputs))
    
class boundary_layer(tf.keras.layers.Layer):

    def __init__(self, units=Config.unit_num):
        super(boundary_layer, self).__init__()
        self.units = units

    def boundary(self, x):
        return tf.nn.relu(tf.math.pow(tf.math.reduce_prod(x, axis=1), tf.cast(tf.shape(x)[-1], tf.float32)))
        
    def call(self, inputs, layer):
        return tf.squeeze(layer(inputs)) - self.boundary(inputs[:,:-1])
        
class constraint_layer(tf.keras.layers.Layer):

    def __init__(self, units=Config.unit_num):
        super(constraint_layer, self).__init__()
        self.units = units

    def boundary(self, x):
        return tf.nn.relu(tf.math.pow(tf.math.reduce_prod(x, axis=1), tf.cast(tf.shape(x)[-1], tf.float32)))
    
    def call(self, inputs, layer):
        return tf.nn.relu(self.boundary(inputs[:,:-1]) - tf.squeeze(layer(inputs)))

        

def custom_loss(y_true, y_pred):
    return tf.reduce_sum(tf.math.square(y_pred))


num1, num2 = int(Config.batch_size * Config.proportion1 / Config.unit_num)*Config.unit_num, int(Config.batch_size * Config.proportion2 / Config.unit_num)*Config.unit_num
num3 = Config.batch_size - num1 - num2
num1, num2, num3 = 128, 128, 128
input1 = tf.keras.Input(shape=(Config.dim + 1,), batch_size=num1)
input2 = tf.keras.Input(shape=(Config.dim + 1,), batch_size=num2)
input3 = tf.keras.Input(shape=(Config.dim + 1,), batch_size=num2)
layer = lstm_layer()
output1 = dynamic_layer()(input1,layer)
output2 = boundary_layer()(input2, layer)
output3 = constraint_layer()(input3, layer)
model = tf.keras.Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

for i in range(Config.epoch_num):
    data1 = tf.random.uniform(shape=(num1, Config.dim + 1), maxval=[10 for i in range(Config.dim)] + [Config.T])
    data2 = tf.random.uniform(shape=(num2, Config.dim + 1), maxval=[10 for i in range(Config.dim)] + [Config.T])
    data3 = tf.random.uniform(shape=(num3, Config.dim + 1), minval=[0 for i in range(Config.dim)] + [Config.T], maxval=[10 for i in range(Config.dim)] + [Config.T])
    with tf.GradientTape() as tape:
        loss = custom_loss(0.0, model([data1, data2, data3], training=True))
    grads = tape.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))


# assume (t,x) are uniformly distributed
