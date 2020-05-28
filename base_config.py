class config:
    def __init__(self):
        """parameters of NN"""
        # parameters of PDE
        self._r = 0.0
        self._sig, self._mu = 0.25, self._r - 0.02
        self.K, self.rho, self.x0 = 1, 0.75, 1
        self.dim, self.T = 20, 2
        # parameters in NN
        self.func_name = 'tanh' # activation function type
        self.batch_size = 512 # batch size
        self.epoch_num = 1000 # number of epochs
        self.mc_num = 448 # times to approximate the second derivative of f in Monte Carlo
        self.delta = 1e-5 # step size in approximate second derivatives of f
        self.unit_num = 50 # number of sample in the input layer
        self.layer_num = 3 # times to repeat LSTM-like layers when approximating solutions of PDE
        self.xmax = 2 # maximum absolute value when generating data
        self.filepath = '/home/alanmei/Alan/Columbia University/Research/' # working directory

Config = config()
    


