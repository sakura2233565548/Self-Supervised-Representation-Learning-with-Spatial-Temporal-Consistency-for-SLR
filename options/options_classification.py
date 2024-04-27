
class opts_SLR_cross_subject():

    def __init__(self):
        self.num_class = 128

        # feeder
        self.train_feeder_args = {
            'data_path': '/data',
            'num_frame_path': '/data',
            'l_ratio': [0.5, 1.0],
            'input_size': 32,
            'data_split': 'train'
        }

        self.test_feeder_args = {
            'data_path': '/data',
            'num_frame_path': '/data',
            'l_ratio': [0.95],
            'input_size': 32,
            'data_split': 'test'
        }