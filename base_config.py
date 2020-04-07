class config:
    def __init__(self):
        """parameters of NN"""
        # parameters of PDE
        self._r = 0.01
        self._sig, self._mu = 0.25, self._r - 0.02
        self.K, self.rho, self.x0 = 1, 0.75, 1
        self.dim, self.T = 3, 2
        # parameters in NN
        self.func_name = 'tanh' # activation function type
        self.batch_size = 1024 # batch size
        self.epoch_num = 1000 # number of epochs
        self.mc_num = 512 # times to approximate the second derivative of f in Monte Carlo
        self.delta = 1e-4 # step size in approximate second derivatives of f
        self.unit_num = 1 # number of sample in the input layer
        self.layer_num = 3 # times to repeat LSTM-like layers when approximating solutions of PDE
        self.xmax = 5 # maximum absolute value when generating data
        self.filepath = '/home/alanmei/Alan/Columbia University/Research/' # working directory

Config = config()
    


