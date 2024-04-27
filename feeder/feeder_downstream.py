# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
from feeder.SLR_Dataset import datasets

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder_SLR(torch.utils.data.Dataset):

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 data_split,
                 data_ratio=None,
                 mmap=True,
                 subset_name=None,
                 num_class=None
                 ):
        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.input_representation = input_representation
        self.crop_resize = True
        self.l_ratio = l_ratio
        self.data_split = data_split
        self.data_ratio = data_ratio
        self.subset_name = subset_name
        assert len(subset_name) is not None
        assert num_class is not None

        self.ds = datasets.TotalDataset(subset_name=subset_name, mask_frames=False, data_split=data_split,
                                        wlasl_class_num=num_class if 'WLASL' in subset_name else 2000,
                                        msasl_class_num=num_class if 'MS_ASL' in subset_name else 1000)
        self.N = self.ds.len()
        if data_ratio is not None:
            self.random_select_data(data_ratio)
        if data_split == 'train':
            self.temporal_crop = augmentations.TemporalRandomCrop(size=input_size, interval=2)
        else:
            self.temporal_crop = augmentations.TemporalCenterCrop(size=input_size, interval=2)
        print('sample length is ', self.__len__())
        print("l_ratio", self.l_ratio)
        print('subset name:', subset_name)
        print('num_class:', num_class)


    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def random_select_data(self, data_ratio):
        idx = np.arange(self.N)
        np.random.shuffle(idx)

        N_used = int(self.N * data_ratio)
        idx_used = idx[:N_used]

        self.N = N_used
        self.index = idx_used

    def collect_data(self, data_numpy, data_raw, indices):
        results = {}
        rh = data_numpy[:, :, :21, 0].transpose(1, 2, 0).astype('float32')
        lh = data_numpy[:, :, 21:42, 0].transpose(1, 2, 0).astype('float32')
        body = data_numpy[:, :, 42:, 0].transpose(1, 2, 0).astype('float32')
        if self.data_split == 'train':
            seed = np.random.randint(low=0, high=10)
            if seed % 2 == 0:
                results['rh'] = rh
                results['lh'] = lh
                results['body'] = body
                rh_mask = data_raw['right']['mask'][indices, :, :].numpy()
                lh_mask = data_raw['left']['mask'][indices, :, :].numpy()
                results['mask'] = np.concatenate([rh_mask, lh_mask], axis=0)
            else:
                results['lh'] = rh
                results['rh'] = lh
                results['body'] = body
                rh_mask = data_raw['right']['mask'][indices, :, :].numpy()
                lh_mask = data_raw['left']['mask'][indices, :, :].numpy()
                results['mask'] = np.concatenate([lh_mask, rh_mask], axis=0)
        else:
            results['rh'] = rh
            results['lh'] = lh
            results['body'] = body
            rh_mask = data_raw['right']['mask'][indices, :, :].numpy()
            lh_mask = data_raw['left']['mask'][indices, :, :].numpy()
            results['mask'] = np.concatenate([rh_mask, lh_mask], axis=0)
        return results

    def __getitem__(self, index):
        if self.data_ratio is not None:
            index = self.index[index]
        data_raw = self.ds.get_sample(index)
        if self.data_split == 'train' and 'WLASL' not in self.subset_name:
            seed = np.random.randint(low=0, high=10)
            if seed % 2 == 0:
                data_numpy = torch.cat([data_raw['right']['kp2d'], data_raw['left']['kp2d'], data_raw['body']['body_pose']], dim=1)
            else:
                data_numpy = torch.cat([data_raw['left']['kp2d'], data_raw['right']['kp2d'], data_raw['body']['body_pose']], dim=1)
        else:
            data_numpy = torch.cat([data_raw['right']['kp2d'], data_raw['left']['kp2d'], data_raw['body']['body_pose']], dim=1)
        data_numpy = data_numpy[:, :, :, None].permute(2, 0, 1, 3).numpy()
        # get raw input

        # input: C, T, V, M
        # data_numpy = np.array(self.data[index])
        number_of_frames = min(data_numpy.shape[1], 300)  # 300 is max_len


        # temporal crop-resize
        data_numpy_crop, indices = augmentations.crop_subsequence(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        label = data_raw['right']['label']


        data_v1 = self.collect_data(data_numpy_crop, data_raw, indices)

        return data_v1, label.long()
