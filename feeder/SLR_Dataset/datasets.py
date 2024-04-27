from feeder.single_dataset.NMFs_CSL import NMFs_CSL
from feeder.single_dataset.MSASL import MSASL
from feeder.single_dataset.WLASL import WLASL
from feeder.single_dataset.SLR500 import SLR500
import torch.utils.data


class TotalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_split='train',
                 data_root='/data',
                 subset_name=['SLR500', 'MS_ASL', 'WLASL', 'NMFs_CSL'],
                 frames=65,
                 threshold=0.4,
                 interval=2,
                 hand_side='right',
                 msasl_class_num=1000,
                 wlasl_class_num=2000,
                 ds_ratio=1.0
                 ):
        self.data_split = data_split
        self.data_root = data_root
        self.frames = frames
        self.datasets = []
        self.threshold = threshold
        self.interval = interval
        self.hand_side = hand_side
        self.ds_ratio = ds_ratio

        dataset_to_index = {}

        '''load SLR500'''
        if 'SLR500' in subset_name:
            self.SLR500 = SLR500(
                data_split=self.data_split,
                interval=self.interval,
                threshold=self.threshold,
                augment=self.aug
            )
            print(self.SLR500)
            dataset_to_index['SLR500'] = self.SLR500

        '''load NMFs_CSL'''
        if 'NMFs_CSL' in subset_name:
            self.NMFs = NMFs_CSL(
                data_split=self.data_split,
                interval=interval,
            )
            print(self.NMFs)
            dataset_to_index['NMFs_CSL'] = self.NMFs

        '''load MS_ASL'''
        if 'MS_ASL' in subset_name:
            self.MS_ASL = MSASL(
                data_split=self.data_split,
                interval=interval,
                class_num=msasl_class_num,
            )
            print(self.MS_ASL)
            dataset_to_index['MS_ASL'] = self.MS_ASL

        '''load WLASL'''
        if 'WLASL' in subset_name:
            self.WLASL = WLASL(
                data_split=self.data_split,
                interval=interval,
                subset_num=wlasl_class_num,
            )
            print(self.WLASL)
            dataset_to_index['WLASL'] = self.WLASL

        for name in subset_name:
            self.datasets.append(dataset_to_index[name])

        self.total_data = 0
        for ds in self.datasets:
            self.total_data += int(len(ds) * self.ds_ratio)

    def get_sample(self, index):
        sample, ds = self._get_sample(index)
        return sample

    def _get_sample(self, index):
        base = 0
        dataset = None
        for ds in self.datasets:
            if index < base + int(len(ds) * self.ds_ratio):
                sample = ds.get_sample(index - base)
                dataset = ds
                break
            else:
                base += int(len(ds) * self.ds_ratio)
        return sample, dataset

    def len(self):
        return self.total_data



