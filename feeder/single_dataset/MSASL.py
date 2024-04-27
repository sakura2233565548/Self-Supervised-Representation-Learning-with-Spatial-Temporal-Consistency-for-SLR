import numpy as np
import os
import torch.utils.data
from termcolor import colored
from tqdm import tqdm
import copy
import math
import pickle as pkl
from PIL import Image

img_size = np.array([[256.0, 256.0]], dtype=np.float32)
LEN_KPS = 10
THRE = 0.8
train_abort_video = [2148, 9478, 9832, 10867, 14378, 14381, 14382, 15055]
cache_home = 'your_path' # you can save pose data into .pkl file


class MSASL(torch.utils.data.Dataset):
    def __init__(self,
                 data_root='/data/MS-ASL',
                 data_split='train',
                 hand_side='right',
                 interval=2,
                 threshold=0.5,
                 max_frames=150,
                 joints=21,
                 class_num=1000,
                 use_cache=True,
                 ):
        '''
        MSASL dataset pre-processing for Model-Aware Transformer
        :param data_root: MSASL Dataset Root Path
        :param data_split: choose from ['train', 'test']
        :param hand_side: if you choose single hand process, this option is useful; else useless
        :param interval: sample interval if video sequence exceeds max_frames
        :param threshold: the threshold for available confidence joints
        :param max_frames: the max frames for single sequence, if exceed this number, sampling will be executed
        :param visualize: if visualize is True, the images will convey to network to visualize the result
        :param joints: the number joints for each hand
        :param class_num: you can choose subset of the whole dataset according to its class number
        '''
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)
        self.name = 'MSASL'
        self.data_root = data_root
        self.data_split = data_split
        self.interval = interval
        self.max_frames = max_frames
        self.video_list = []
        self.annotation = []
        self.threshold = threshold
        self.hand_side = hand_side
        self.joints = joints
        self.use_cache = use_cache
        subset_num = class_num
        if self.data_split == 'train':
            video_list_train = np.genfromtxt(os.path.join(self.data_root, 'traintestlist', 'trainlist01.txt'), delimiter=' ',
                                       dtype=str).tolist()
            for i in range(len(train_abort_video)):
                del video_list_train[train_abort_video[i]-i]
            video_list_train = self.get_subset(video_list_train, subset_num)
            video_list_val = np.genfromtxt(os.path.join(self.data_root, 'traintestlist', 'vallist01.txt'), delimiter=' ',
                                       dtype=str).tolist()
            video_list_val = self.get_subset(video_list_val, subset_num)
            video_list = video_list_train + video_list_val
            video_list = np.array(video_list)
            self.flag = len(video_list_train)
            self.annotation_path = os.path.join(self.data_root, 'Keypoints_2d_mmpose')
        elif self.data_split == 'test':
            video_list = np.genfromtxt(os.path.join(self.data_root, 'traintestlist', 'testlist01.txt'), delimiter=' ',
                                       dtype=str).tolist()
            video_list = self.get_subset(video_list, subset_num)
            video_list = np.array(video_list)
            self.annotation_path = os.path.join(self.data_root, 'Keypoints_2d_mmpose/test')
        if self.use_cache:
            with open(os.path.join(cache_home, f"{self.name}_{self.data_split}.pkl"), 'rb') as f:
                self.pkl_data = pkl.load(f)
        self.video_list = video_list
        print(len(self.video_list))

    def __len__(self):
        return len(self.video_list)

    def __str__(self):
        info = "MSASL {} set. lenth {}".format(
            self.data_split, len(self.video_list)
        )
        return colored(info, 'blue', attrs=['bold'])

    def get_subset(self, video_list, subset_num):
        '''
        :param video_list: the origin dataset list
        :param subset_num: the class number for subset
        :return: new data list for subset
        '''
        video_list_new = []
        for i in range(len(video_list)):
            if int(video_list[i][1]) < subset_num:
                video_list_new.append(video_list[i])
        return video_list_new

    def extractvideoinfo(self, index, video_name):
        if self.use_cache:
            video_data = copy.deepcopy(self.pkl_data[video_name]['video_data'])
            ori_img_size = copy.deepcopy(self.pkl_data[video_name]['img_size'])
        else:
            if self.data_split == 'train':
                if index < self.flag:
                    data_split = 'train'
                else:
                    data_split = 'val'
            else:
                data_split = 'test'
            sample_img = os.path.join(self.data_root, 'jpg_video_ori', data_split, video_name.split('.')[0],
                                      'img_00001.jpg')
            frames_path = os.path.join(self.data_root, 'jpg_video_ori', data_split, video_name.split('.')[0])
            img = Image.open(sample_img)
            ori_img_size = np.array([[img.width, img.height]], dtype=np.float32)
            if self.data_split == 'train':
                json_path = os.path.join(self.annotation_path, data_split, video_name.split('.')[0] + '.pkl')
            else:
                json_path = os.path.join(self.annotation_path, video_name.split('.')[0] + '.pkl')
            video_data = pkl.load(open(json_path, 'rb'))
        return video_data, ori_img_size


    def get_single_hand(self, index, total_frame_list):
        '''
        :param index: the index of video
        :return: dict type, single hand information
        '''
        video_index = self.video_list[index]
        video_name = video_index[0]
        label = int(video_index[1])
        video_data, ori_img_size = self.extractvideoinfo(index, video_name)
        video_joints = video_data['keypoints']
        frame_list = video_data['img_list']
        if len(frame_list) > self.max_frames:
            frame_index = slice(0, len(video_joints), max(self.interval, math.ceil(len(frame_list) / 150)))
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]
        video_joints_update, bbxes, rescale_bbx, frame_list_update = self.crop_hand(video_joints, ori_img_size,
                                                                                    frame_list)
        if len(total_frame_list) == 0 or video_joints_update is None:
            kp2ds_total = torch.zeros((len(total_frame_list), 21, 2), dtype=torch.float32)
            conf = torch.zeros_like(kp2ds_total)
            gts = torch.zeros_like(kp2ds_total)
            clrs = []
            scale = torch.from_numpy(img_size).float()
            label = torch.tensor(label, dtype=torch.int)
            sample = {
                'clr': clrs,
                'kp2d': kp2ds_total,
                'flag_2d': conf,
                'label': label,
                'gts': gts,
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
            skeleton = video_joints_update[i][:, 0:2]
            confidences = video_joints_update[i][:, 2]
            kp2ds, confidence = self.get_kp2ds(skeleton, confidences, self.threshold, part=self.hand_side)
            mask = copy.deepcopy(confidence)
            mask = np.where(mask > self.threshold, 1.0, 0.0)
            trans = np.array([[bbxes[i][0], bbxes[i][2]]], dtype=np.float32)
            scale = np.array(
                [[bbxes[i][1] - bbxes[i][0], bbxes[i][3] - bbxes[i][2]]],
                dtype=np.float32)
            if scale[0, 1] == 0.0 or scale[0, 0] == 0:
                print('wrong!')
            assert scale[0, 1] > 0.0 and scale[0, 0] > 0.0
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

        # existing condition that only last several frames don't exist
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
            'scale': scale,
        }
        return sample

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

    def get_bbox(self, kp2ds, conf, scale_rate=1.0):
        # kp2ds = kp2ds[np.where(conf>self.threshold), :].squeeze(0)

        x_max, y_max = np.max(kp2ds, axis=0)
        x_min, y_min = np.min(kp2ds, axis=0)
        x_center, y_center = (x_min+x_max) / 2.0, (y_min+y_max) / 2.0
        scale = max(x_max-x_min, y_max-y_min) * scale_rate
        center = np.array([x_center, y_center], dtype=np.float32)
        return center, scale

    def get_body_pose(self, index, total_frame_list):
        video_index = self.video_list[index]
        video_name = video_index[0]
        video_data, ori_img_size = self.extractvideoinfo(index, video_name)
        video_joints = video_data['keypoints']
        frame_list = video_data['img_list']


        root_pos = []
        root_pos_gt = []
        root_pos_valid = []
        root_mask = []
        for frame in total_frame_list:
            i = frame_list.index(frame)
            skeleton = video_joints[i][:, 0:2]
            confidences = video_joints[i][:, 2]
            body_pose, body_conf = self.get_kp2ds(skeleton, confidences, 0.0, part='body')
            gt = copy.deepcopy(body_pose) / ori_img_size * img_size
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
        video_index = self.video_list[index]
        video_name = video_index[0]
        video_data, ori_img_size = self.extractvideoinfo(index, video_name)
        video_joints = video_data['keypoints']
        frame_list = video_data['img_list']
        if len(frame_list) > self.max_frames:
            frame_index = slice(0, len(video_joints), max(self.interval, math.ceil(len(frame_list) / 150)))
            video_joints = video_joints[frame_index, :, :]
            frame_list = frame_list[frame_index]
        self.hand_side = 'right'
        _, _, _, frame_list_update_right = self.crop_hand(video_joints, ori_img_size,frame_list)
        self.hand_side = 'left'
        _, _, _, frame_list_update_left = self.crop_hand(video_joints, ori_img_size, frame_list)
        if frame_list_update_left is None:
            total_frame_list = frame_list_update_right
        elif frame_list_update_right is None:
            total_frame_list = frame_list_update_left
        else:
            total_frame_list = list(set(frame_list_update_right + frame_list_update_left))
        total_frame_list = sorted(total_frame_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        return total_frame_list

    def get_kp2ds(self, skeleton, conf, threshold, part):
        '''
        extract hands keypoints from mmpose skeleton points
        :param skeleton: mmpose skeleton points sequence
        :param conf: the confidence of skeleton points
        :param threshold: the threshold of available joints
        :return: single hand sequence and their confidence
        '''
        if part == 'left':
            hand_kp2d = skeleton[91:112, :]
            confidence = conf[91:112]
        elif part == 'right':
            hand_kp2d = skeleton[112:133, :]
            confidence = conf[112:133]
        elif part == 'body':
            hand_kp2d = skeleton[[0,5,7,9,6,8,10], :]
            confidence = conf[[0,5,7,9,6,8,10]]
        else:
            raise Exception('wrong hand_side type')
        confidence = np.where(confidence > threshold, confidence, 0.0)
        indexes = np.where(confidence < threshold)[0].tolist()
        for i in range(len(indexes)):
            hand_kp2d[indexes[i]] = np.zeros((1, 2), dtype=np.float32)
        confidence = np.tile(confidence[:, np.newaxis], (1, 2))
        return hand_kp2d, confidence


    def crop_hand(self, frames_list, ori_img_size, frames_name):
        frames_list_new = []
        frames_name_new = []
        bbxes = []
        for i in range(len(frames_name)):
            skeleton = frames_list[i, :, 0:2]
            confidence = frames_list[i, :, 2]
            usz, vsz = [ori_img_size[0][0], ori_img_size[0][1]]
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
                bbx = self.elbow_hand(skeleton, confidence, ori_img_size)
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

    def elbow_hand(self, pose_keypoints, confidence, ori_img_size):
        right_hand = pose_keypoints[[6, 8, 10]]
        left_hand = pose_keypoints[[5, 7, 9]]
        ratioWristElbow = 0.33
        detect_result = []
        usz, vsz = [ori_img_size[0][0], ori_img_size[0][1]]
        if self.hand_side == 'right':
            has_right = np.sum(confidence[[6, 8, 10]] < THRE) == 0 and right_hand[2, 1] < vsz - 50
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
            has_left = np.sum(confidence[[5, 7, 9]] < THRE) == 0 and left_hand[2, 1] < vsz - 50
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
            video_index = self.video_list[index]
            video_name = video_index[0]
            label = int(video_index[1])
            if self.data_split == 'train':
                if index < self.flag:
                    data_split = 'train'
                else:
                    data_split = 'val'
            else:
                data_split = 'test'
            sample_img = os.path.join(self.data_root, 'jpg_video_ori', data_split, video_name.split('.')[0],
                                      'img_00001.jpg')
            img = Image.open(sample_img)
            ori_img_size = np.array([[img.width, img.height]], dtype=np.float32)
            if self.data_split == 'train':
                json_path = os.path.join(self.annotation_path, data_split, video_name.split('.')[0] + '.pkl')
            else:
                json_path = os.path.join(self.annotation_path, video_name.split('.')[0] + '.pkl')
            video_data = pkl.load(open(json_path, 'rb'))
            sample = {"video_name": video_name, 'video_data':video_data, 'label': label, 'img_size': ori_img_size}
            pkl_data[video_name] = sample
        with open(os.path.join(cache_home, f"{self.name}_{self.data_split}.pkl"), 'wb') as f:
            pkl.dump(pkl_data, f)
            print(f"save pkl data to {cache_home}!")
