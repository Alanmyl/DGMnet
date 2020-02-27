class config:
    def __init__(self):
        self._r = 0
        self._sig, self._mu = 0.25, self._r - 0.02
        self.K, self.rho, self.x0 = 1, 0.75, 1
        self.dim, self.T = 20, 2
        self.func_name = 'tanh'
        self.batch_size = 128
        self.epoch_num = 30
        self.mc_num = 500
        self.delta = 1e-3
        self.unit_num = 16
        self.layer_num = 3
        self.proportion1 = 0.5
        self.proportion2 = 0.3

Config = config()
    


