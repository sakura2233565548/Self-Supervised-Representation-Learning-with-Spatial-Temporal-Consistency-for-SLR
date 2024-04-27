import torch.nn.functional as F
import torch
import random
import numpy as np


def joint_courruption(input_data):                                                                     

    out = input_data.copy()

    flip_prob = random.random()

    if flip_prob < 0.5:

        #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(49, 25,replace=False)
        out[:,:,joint_indicies,:] = 0 
        return out
    
    else:
         #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
         joint_indicies = np.random.choice(49, 25,replace=False)
         
         temp = out[:,:,joint_indicies,:] 
         # Corruption = np.array([
         #                   [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
         #                   [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
         #                   [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] ])
         Corruption = np.array([
                           [random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1)]])
         temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
         temp = temp.transpose(3, 0, 1, 2)
         out[:,:,joint_indicies,:] = temp
         return out



def pose_augmentation(input_data):


        # Shear       = np.array([
        #               [1,	random.uniform(-1, 1), 	random.uniform(-1, 1)],
        #               [random.uniform(-1, 1), 1, 	random.uniform(-1, 1)],
        #               [random.uniform(-1, 1), 	random.uniform(-1, 1),      1]
        #               ])
        Shear = np.array([
            [1, random.uniform(-1, 1)],
            [random.uniform(-1, 1), 1]
        ])

        temp_data = input_data.copy()
        result =  np.dot(temp_data.transpose([1, 2, 3, 0]),Shear.transpose())
        output = result.transpose(3, 0, 1, 2)

        return output

def temporal_cropresize(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    # Temporal crop
    min_crop_length = 32
    # min_crop_length = 64

    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

    start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:,start:start+temporal_crop_length, :, :]

    indices = [int(i * temporal_crop_length / output_size) + start for i in range(output_size)]


    # interpolate
    temporal_context = torch.tensor(temporal_context,dtype=torch.float)
    temporal_context=temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
    temporal_context=temporal_context[None, :, :, None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0) 
    temporal_context=temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context, indices

class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, interval):
        self.size = size * interval
        self.interval = interval

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        out = out[::self.interval]
        return out

def crop_subsequence(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    if l_ratio[0] == 0.5:
    # if training , sample a random crop

         min_crop_length = 64
         scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
         temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

         start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
         temporal_crop = input_data[:,start:start+temporal_crop_length, :, :]

         temporal_crop= torch.tensor(temporal_crop,dtype=torch.float)
         temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
         temporal_crop=temporal_crop[None, :, :, None]
         temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
         temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
         temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

         indices = [int(i * temporal_crop_length / output_size) + start for i in range(output_size)]

         return temporal_crop, indices

    else:
    # if testing , sample a center crop

        start = int((1-l_ratio[0]) * num_of_frames/2)
        data =input_data[:,start:num_of_frames-start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop= torch.tensor(data,dtype=torch.float)
        temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
        temporal_crop=temporal_crop[None, :, :, None]
        temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
        temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
        temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        indices = [int(i * temporal_crop_length / output_size) + start for i in range(output_size)]

        return temporal_crop, indices

class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, interval):
        self.size = size * interval
        self.interval = interval

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        out = out[::self.interval]

        return out
