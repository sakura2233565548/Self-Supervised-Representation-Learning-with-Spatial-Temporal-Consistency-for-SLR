import numpy as np
import os
import torch.utils.data
import random
import pickle as pkl
import json
from termcolor import colored
from tqdm import tqdm
import copy
import math
from PIL import Image

img_size = np.array([[256.0, 256.0]], dtype=np.float32)
img_width = 1280
img_height = 720
ori_img_size = np.array([[img_width, img_height]], dtype=np.float32)
LEN_KPS = 10
THRE = 0.5
cache_home = 'your_path' # you can save pose data into .pkl file

class SLR500(torch.utils.data.Dataset):
    def __init__(self,
                 data_root='/data/SLR500',
                 data_split='train',
                 hand_side='left',
                 interval=2,
                 threshold=0.5,
                 max_frames=150,
                 joints=21,
                 use_cache=True,
                 ):
        '''
        SLR500 dataset pre-processing for Model-Aware Transformer
        :param data_root: SLR500 Dataset Root Path
        :param data_split: choose from ['train', 'test']
        :param hand_side: if you choose single hand process, this option is useful; else useless
        :param interval: sample interval if video sequence exceeds max_frames
        :param threshold: the threshold for available confidence joints
        :param max_frames: the max frames for single sequence, if exceed this number, sampling will be executed
        :param joints: the number joints for each hand
        '''
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)
        self.name = 'SLR500'
        self.data_root = data_root
        self.data_split = data_split
        self.interval = interval
        self.max_frames = max_frames
        self.video_list = []
        self.annotation = []
        self.threshold = threshold
        self.hand_side = hand_side
        self.use_cache = use_cache
        self.joints = joints
        if self.data_split == 'train':
            video_list = open(os.path.join(self.data_root, 'traintestlist', 'trainlist01.txt'), 'r').readlines()
        elif self.data_split == 'test':
            video_list = open(os.path.join(self.data_root, 'traintestlist', 'testlist01.txt'), 'r').readlines()
        video_list = [video.split('.')[0] for video in video_list]
        self.video_list = video_list
        self.visual_path = os.path.join(self.data_root, 'jpg_video_trim')
        self.pkl_path = os.path.join(self.data_root, 'Keypoints_2d_mmpose')
        if self.use_cache:
            with open(os.path.join(cache_home, f"{self.name}_{self.data_split}.pkl"), 'rb') as f:
                self.pkl_data = pkl.load(f)

    def __len__(self):
        return len(self.video_list)

    def __str__(self):
        info = "SLR500 {} set. length {}".format(
            self.data_split, len(self.video_list)
        )
        return colored(info, 'blue', attrs=['bold'])

    def extractvideoinfo(self, index, video_name):
        if self.use_cache:
            video_data = copy.deepcopy(self.pkl_data[video_name]['video_data'])
        else:
            video_data = pkl.load(open(os.path.join(self.pkl_path, video_name + '.pkl'), 'rb'))
        return video_data

    def get_single_hand(self, index, total_frame_list):
        '''
        :param index: the index of video
        :return: dict type, single hand information
        '''
        video_name = self.video_list[index]
        label = int(video_name.split('/')[0])
        pkl_data = self.extractvideoinfo(index, video_name)
        video_joints = pkl_data['keypoints']
        frame_list = pkl_data['img_list'][:-1]

        # sample frames
        if len(frame_list) > self.max_frames:
            frame_index = slice(0, len(video_joints), max(self.interval, math.ceil(len(frame_list) / 150)))
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]

        # choose useful frames, delete invalid frames
        video_joints_update, bbxes, rescale_bbx, frame_list_update = self.crop_hand(video_joints, frame_list)

        if len(total_frame_list) == 0 and video_joints_update is None:
            kp2ds_total = torch.zeros((len(total_frame_list), 21, 2), dtype=torch.float32)
            conf = torch.zeros_like(kp2ds_total)
            gts = torch.zeros_like(kp2ds_total)
            masks = torch.zeros_like(kp2ds_total)
            clrs = []
            scale = torch.from_numpy(img_size).float()
            label = torch.tensor(label, dtype=torch.int)
            sample = {
                'clr': clrs,
                'kp2d': kp2ds_total,
                'flag_2d': conf,
                'label': label,
                'gts': gts,
                'mask': masks,
                'scale': scale
            }
            return sample


        kp2ds_total = []
        conf = []
        gts = []
        masks = []
        clrs = []
        for i in range(len(video_joints_update)):
            frame_id = total_frame_list.index(frame_list_update[i])
            if frame_id != i:
                del total_frame_list[i:frame_id]
                length = frame_id - i
                kp2d = torch.zeros((length, 21, 2), dtype=torch.float32)
                kp2ds_total.append(kp2d)
                conf.append(torch.zeros_like(kp2d))
                gts.append(torch.zeros_like(kp2d))
                masks.append(torch.zeros_like(kp2d))
            kp2d = video_joints_update[i]
            kp2ds, confidence = self.get_kp2ds(kp2d, threshold=self.threshold, part=self.hand_side)
            mask = copy.deepcopy(confidence)
            mask = np.where(mask > self.threshold, 1.0, 0.0)
            trans = np.array([[bbxes[i][0], bbxes[i][2]]], dtype=np.float32)
            scale = np.array(
                [[bbxes[i][1] - bbxes[i][0], bbxes[i][3] - bbxes[i][2]]],
                dtype=np.float32)
            kp2ds = (kp2ds - trans) / scale * img_size
            kp2ds = np.where(kp2ds > 0.0, kp2ds, 0.0)
            if self.hand_side == 'left':
                kp2ds[:, 0] = img_size[0, 0] - kp2ds[:, 0]
                kp2ds = np.where(kp2ds < 255.0, kp2ds, 0.0)
            gt = copy.deepcopy(kp2ds)


            kp2ds = kp2ds / img_size
            kp2ds_total.append(kp2ds[np.newaxis, :, :])
            conf.append(confidence[np.newaxis, :, :])
            gts.append(gt[np.newaxis, :, :])
            masks.append(mask[np.newaxis, :, :])

        # existing condition that only last several frames don;t exist
        if len(total_frame_list) != len(frame_list_update):
                length = len(total_frame_list) - len(frame_list_update)
                kp2d = torch.zeros((length, 21, 2), dtype=torch.float32)
                kp2ds_total.append(kp2d)
                conf.append(torch.zeros_like(kp2d))
                gts.append(torch.zeros_like(kp2d))
                masks.append(torch.zeros_like(kp2d))

        kp2ds_total = np.concatenate(kp2ds_total, axis=0).astype(np.float32)
        conf = np.concatenate(conf, axis=0).astype(np.float32)
        gts = np.concatenate(gts, axis=0).astype(np.float32)
        masks = np.concatenate(masks, axis=0).astype(np.float32)
        kp2ds_total = torch.from_numpy(kp2ds_total).float()
        conf = torch.from_numpy(conf).float()
        gts = torch.from_numpy(gts).float()
        masks = torch.from_numpy(masks).float()
        scale = torch.from_numpy(img_size).float()
        label = torch.tensor(label, dtype=torch.int)
        sample = {
            'clr': clrs,
            'kp2d': kp2ds_total,
            'flag_2d': conf,
            'label': label,
            'gts': gts,
            'mask': masks,
            'scale': scale
        }
        return sample

    def get_bbox(self, kp2ds, conf, scale_rate=1.0):
        # kp2ds = kp2ds[np.where(conf>self.threshold), :].squeeze()

        x_max, y_max = np.max(kp2ds, axis=0)
        x_min, y_min = np.min(kp2ds, axis=0)
        x_center, y_center = (x_min+x_max) / 2.0, (y_min+y_max) / 2.0
        scale = max(x_max-x_min, y_max-y_min) * scale_rate
        center = np.array([x_center, y_center], dtype=np.float32)
        return center, scale

    def get_sample(self, index):
        '''
        :param index:
        :return: dict type, both right and left hands
        '''
        hand_sides = ['right', 'left']
        total_frame_list = self.GetTotalFrameList(index)
        sample = {}
        for hand_side in hand_sides:
            self.hand_side = hand_side
            sample[hand_side] = self.get_single_hand(index, total_frame_list.copy())
        sample['body'] = self.get_body_pose(index, total_frame_list)
        assert sample['right']['kp2d'].shape[0] == sample['left']['kp2d'].shape[0]
        return sample

    def get_body_pose(self, index, total_frame_list):
        video_name = self.video_list[index]
        pkl_data = self.extractvideoinfo(index, video_name)
        video_joints = pkl_data['keypoints']
        frame_list = pkl_data['img_list'][:-1]


        root_pos = []
        root_pos_gt = []
        root_pos_valid = []
        root_mask = []
        for frame in total_frame_list:
            i = frame_list.index(frame)
            kp2d = video_joints[i]
            body_pose, body_conf = self.get_kp2ds(kp2d, threshold=0.0, part='body')
            gt = body_pose / ori_img_size * img_size
            mask = copy.deepcopy(body_conf)
            mask = np.where(mask > self.threshold, 1.0, 0.0)
            body_pose = body_pose / ori_img_size
            root_pos.append(body_pose)
            root_pos_gt.append(gt)
            root_pos_valid.append(body_conf)
            root_mask.append(mask)

        root_pos = np.stack(root_pos, axis=0).astype(np.float32)
        root_pos_gt = np.stack(root_pos_gt, axis=0).astype(np.float32)
        root_pos_valid = np.stack(root_pos_valid, axis=0).astype(np.float32)
        root_mask = np.stack(root_mask, axis=0).astype(np.float32)
        root_pos = torch.from_numpy(root_pos).float()
        root_pos_gt = torch.from_numpy(root_pos_gt).float()
        root_pos_valid = torch.from_numpy(root_pos_valid).float()
        root_mask = torch.from_numpy(root_mask).float()
        sample = {
            'body_pose': root_pos,
            'body_pose_gt': root_pos_gt,
            'body_pose_conf': root_pos_valid,
            'root_mask': root_mask
        }
        return sample

    def GetTotalFrameList(self, index):
        video_name = self.video_list[index]
        pkl_data = self.extractvideoinfo(index, video_name)
        video_joints = pkl_data['keypoints']
        frame_list = pkl_data['img_list'][:-1]

        # sample frames
        if len(frame_list) > self.max_frames:
            frame_index = slice(0, len(video_joints), max(self.interval, math.ceil(len(frame_list) / 150)))
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]
        self.hand_side = 'right'
        _, _, _, frame_list_update_right = self.crop_hand(video_joints, frame_list)
        self.hand_side = 'left'
        _, _, _, frame_list_update_left = self.crop_hand(video_joints, frame_list)
        total_frame_list = list(set(frame_list_update_right+frame_list_update_left))
        total_frame_list = sorted(total_frame_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        return total_frame_list

    def get_kp2ds(self, kp2d, part, threshold=0.4):
        '''
        extract hands keypoints from mmpose skeleton points
        :param kp2d: mmpose skeleton points sequence
        :param threshold: the threshold of available joints
        :return: single hand sequence and their confidence
        '''
        if part == 'left':
            kp2ds = kp2d[91:112, :2]
            confidence = kp2d[91:112, 2]
        elif part== 'right':
            kp2ds = kp2d[112:133, :2]
            confidence = kp2d[112:133, 2]
        elif part == 'body':
            kp2ds = kp2d[[0, 5, 7, 9, 6, 8, 10], :2]
            confidence = kp2d[[0, 5, 7, 9, 6, 8, 10], 2]
        else:
            raise Exception('wrong hand_side type')
        confidence = np.where(confidence > threshold, confidence, 0.0)
        abort_indexes = np.where(confidence < threshold)[0].tolist()

        # invalid points set zeroes
        for i in range(len(abort_indexes)):
            kp2ds[abort_indexes[i]] = np.zeros((1, 2), dtype=np.float32)
        confidence = np.tile(confidence[:, np.newaxis], (1, 2))
        return kp2ds, confidence

    def crop_hand(self, frames_list, frames_name):
        frames_list_new = []
        frames_name_new = []
        bbxes = []
        for i in range(len(frames_name)):
            skeleton = frames_list[i, :, 0:2]
            confidence = frames_list[i, :, 2]
            usz, vsz = [img_width, img_height]
            minsz = min(usz, vsz)
            maxsz = max(usz, vsz)
            if self.hand_side == 'right':
                right_keypoints = skeleton[112:133, :]
                kp_visible = (confidence[112:133] > 0.1)
                uvis = right_keypoints[kp_visible, 0]
                vvis = right_keypoints[kp_visible, 1]
            elif self.hand_side == 'left':
                left_keypoints = skeleton[91:112, :]
                kp_visible = (confidence[91:112] > 0.1)
                uvis = left_keypoints[kp_visible, 0]
                vvis = left_keypoints[kp_visible, 1]
            else:
                raise ValueError('wrong hand side')
            if len(uvis) < LEN_KPS:
                bbx = self.elbow_hand(skeleton, confidence)
                if bbx is None:
                    continue
                else:
                    bbxes.append(bbx)
                    frames_list_new.append(frames_list[i])
                    frames_name_new.append(frames_name[i])
            else:
                umin = min(uvis)
                vmin = min(vvis)
                umax = max(uvis)
                vmax = max(vvis)

                B = round(2.2 * max([umax - umin, vmax - vmin]))

                us = 0
                ue = usz - 1
                vs = 0
                ve = vsz - 1
                umid = umin + (umax - umin) / 2
                vmid = vmin + (vmax - vmin) / 2

                if (B < minsz - 1):
                    us = round(max(0, umid - B / 2))
                    ue = us + B
                    if (ue > usz - 1):
                        d = ue - (usz - 1)
                        ue = ue - d
                        us = us - d
                    vs = round(max(0, vmid - B / 2))
                    ve = vs + B
                    if (ve > vsz - 1):
                        d = ve - (vsz - 1)
                        ve = ve - d
                        vs = vs - d
                if (B >= minsz - 1):
                    B = minsz - 1
                    if usz == minsz:
                        vs = round(max(0, vmid - B / 2))
                        ve = vs + B
                        if (ve > vsz - 1):
                            d = ve - (vsz - 1)
                            ve = ve - d
                            vs = vs - d
                    if vsz == minsz:
                        us = round(max(0, umid - B / 2))
                        ue = us + B

                        if (ue > usz - 1):
                            d = ue - (usz - 1)
                            ue = ue - d
                            us = us - d
                us = int(us)
                vs = int(vs)
                ue = int(ue)
                ve = int(ve)
                bbx = [us, ue, vs, ve]
                bbxes.append(bbx)
                frames_list_new.append(frames_list[i])
                frames_name_new.append(frames_name[i])

        bbxes = np.array(bbxes, dtype=np.float32)
        if len(bbxes) == 0:
            return None, None, None, None
        average_width = np.average(bbxes[:, 1] - bbxes[:, 0])
        average_height = np.average(bbxes[:, 3] - bbxes[:, 2])
        rescale_bbx = np.array([average_width, average_height], dtype=np.float32)
        return frames_list_new, bbxes, rescale_bbx, frames_name_new

    def elbow_hand(self, pose_keypoints, confidence):
        right_hand = pose_keypoints[[6, 8, 10]]
        left_hand = pose_keypoints[[5, 7, 9]]
        ratioWristElbow = 0.33
        detect_result = []
        usz, vsz = [img_width, img_height]
        if self.hand_side == 'right':
            has_right = np.sum(confidence[[6, 8, 10]] < THRE) == 0
            if not has_right:
                return None
            x1, y1 = right_hand[0][:2]
            x2, y2 = right_hand[1][:2]
            x3, y3 = right_hand[2][:2]

            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            x -= width / 2
            y -= width / 2  # width = height

            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > usz: width1 = usz - x
            if y + width > vsz: width2 = vsz - y
            width = min(width1, width2)
            detect_result.append([int(x), int(y), int(width)])

        elif self.hand_side == 'left':
            has_left = np.sum(confidence[[5, 7, 9]] < THRE) == 0
            if not has_left:
                return None
            x1, y1 = left_hand[0][:2]
            x2, y2 = left_hand[1][:2]
            x3, y3 = left_hand[2][:2]

            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            x -= width / 2
            y -= width / 2  # width = height
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > usz: width1 = usz - x
            if y + width > vsz: width2 = vsz - y
            width = min(width1, width2)
            detect_result.append([int(x), int(y), int(width)])

        x, y, width = int(x), int(y), int(width)
        return [x, x + width, y, y + width]

    def generatepkl(self):
        pkl_data = {}
        for index in tqdm(range(self.__len__())):
            video_name = self.video_list[index]
            label = int(video_name.split('/')[0])
            video_data = pkl.load(open(os.path.join(self.pkl_path, video_name + '.pkl'), 'rb'))
            sample = {"video_name": video_name, 'video_data': video_data, 'label': label, 'img_size': ori_img_size}
            pkl_data[video_name] = sample
        with open(os.path.join(cache_home, f"{self.name}_{self.data_split}.pkl"), 'wb') as f:
            pkl.dump(pkl_data, f)
            print(f"save pkl data to {cache_home}!")
