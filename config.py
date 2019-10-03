class NABConfig:
    def __init__(self):
        self.T = 12 # timestep
        self.W = 2 # convolution window size (convolution filter height)` ?
        self.n = 7 # the number of the long-term memory series
        self.highway_window = 10  # the window size of ar model

        self.D = 3  # input's variable dimension (convolution filter width)
        self.K = 1 # output's variable dimension

        # self.horizon = 6 # the horizon of predicted value

        self.en_conv_hidden_size = 16
        self.en_rnn_hidden_sizes = [16, 16]  # last size is equal to en_conv_hidden_size 

        self.input_keep_prob = 0.8
        self.output_keep_prob = 1.0

        self.lr = 0.001
        self.batch_size = 100
