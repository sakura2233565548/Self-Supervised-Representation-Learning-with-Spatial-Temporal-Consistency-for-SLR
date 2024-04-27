# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
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
                 mmap=True):
        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.input_representation = input_representation
        self.crop_resize = True
        self.l_ratio = l_ratio

        self.ds = datasets.TotalDataset(subset_name=['NMFs_CSL', 'SLR500', 'MS_ASL', 'WLASL'], mask_frames=False)
        self.temporal_crop = augmentations.TemporalRandomCrop(size=input_size, interval=2)
        print('sample length is ', self.__len__())
        print("l_ratio", self.l_ratio)


    def __len__(self):
        return self.ds.len()

    def __iter__(self):
        return self

    def collect_data(self, data_numpy, data_raw, indices):
        results = {}
        rh = data_numpy[:, :, :21, 0].transpose(1, 2, 0).astype('float32')
        lh = data_numpy[:, :, 21:42, 0].transpose(1, 2, 0).astype('float32')
        body = data_numpy[:, :, 42:, 0].transpose(1, 2, 0).astype('float32')
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
        return results


    def __getitem__(self, index):
        # get raw input
        data_raw = self.ds.get_sample(index)
        data_numpy = torch.cat([data_raw['right']['kp2d'], data_raw['left']['kp2d'], data_raw['body']['body_pose']], dim=1)
        data_numpy = data_numpy[:, :, :, None].permute(2, 0, 1, 3).numpy()

        # input: C, T, V, M
        number_of_frames = min(data_numpy.shape[1], 300)  # 300 is max_len

        # apply spatio-temporal augmentations to generate  view 1

        # temporal crop-resize
        data_numpy_v1_crop, indices_v1 = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations
        flip_prob = random.random()
        if flip_prob < 0.5:
                 data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)
        else:
                 data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1_crop)

        data_v1 = self.collect_data(data_numpy_v1, data_raw, indices_v1)


        # apply spatio-temporal augmentations to generate  view 2

        # temporal crop-resize
        data_numpy_v2_crop, indices_v2 = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)

        data_v2 = self.collect_data(data_numpy_v2, data_raw, indices_v2)

        #View 1
        input_v1 = data_numpy_v1.astype('float32')
        #View 2
        input_v2 = data_numpy_v2.astype('float32')

        return data_v1, data_v2
