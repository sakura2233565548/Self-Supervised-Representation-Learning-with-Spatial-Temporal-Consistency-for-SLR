class opts_SLR_cross_subject():

    def __init__(self):
        self.num_class = 128

        # feeder
        self.train_feeder_args = {
            'data_path': '/data',
            'num_frame_path': '/data',
            'l_ratio': [0.1, 1],
            'input_size': 64
        }