class config:
    def __init__(self):
        self._r = 0
        self._sig, self._mu = 0.25, self._r - 0.02
        self.K, self.rho, self.x0 = 1, 0.75, 1
        self.dim, self.T = 20, 2
        self.func_name = 'tanh'
        self.batch_size = 160
        self.epoch_num = 1050
        self.mc_num = 512
        self.delta = 1e-4
        self.unit_num = 40
        self.layer_num = 3
        self.xmax = 5
        self.filepath = '/home/alanmei/Alan/Columbia University/Research/'

Config = config()
    


