import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN_Transformer import  moco_model_with_transfer, Config

def loss_kld(inputs, targets):
    inputs = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')


def init_para_GCN_Trans(model):
    print("<<< initialize the para in transformer!")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class MoCo(nn.Module):
    def __init__(self, skeleton_representation, num_class, dim=128, K=65536, m=0.999, T=0.07,
                 teacher_T=0.05, student_T=0.1, cmd_weight=1.0, topk=1024, mlp=False, pretrain=True, dropout=None,
                 inter_weight=0.5, inter_dist=False, topk_part=1024, K_part=8192):
        super(MoCo, self).__init__()
        self.pretrain = pretrain
        RHand_Bone = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 1), (7, 6), (8, 7), (9, 8), (10, 1),
                     (11, 10), (12, 11), (13, 12), (14, 1), (15, 14), (16, 15), (17, 16), (18, 1),
                     (19, 18), (20, 19), (21, 20)]
        Body_Bone = [(2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6)]
        self.Bone = RHand_Bone + [(k+21, v+21) for k,v in RHand_Bone] + [(k+42, v+42) for k,v in Body_Bone]
        cfg= Config()
        cfg.proj_dropout = dropout
        cfg.num_class = num_class
        self.encoder_q = moco_model_with_transfer(cfg)
        self.encoder_q_motion = moco_model_with_transfer(cfg)
        init_para_GCN_Trans(self.encoder_q)
        init_para_GCN_Trans(self.encoder_q_motion)



    def forward(self, im_q, im_k=None, view='joint', extract_feat=False, knn_eval=False, self_dist=False):
        im_q_motion = {}
        for k,v in im_q.items():
            if k in ['rh', 'lh', 'body']:
                temp_tensor = torch.zeros_like(v)
                temp_tensor[:, :-1, :, :] = v[:, 1:, :, :] - v[:, :-1, :, :]
                im_q_motion[k] = temp_tensor
            else:
                im_q_motion[k] = v

        if not self.pretrain:
            if extract_feat == False:
                if view == 'joint':
                    return self.encoder_q(im_q, knn_eval)[0]
                elif view == 'motion':
                    return self.encoder_q_motion(im_q_motion, knn_eval)[0]
                elif view == 'all':
                    return (self.encoder_q(im_q, knn_eval)[0] + \
                            self.encoder_q_motion(im_q_motion, knn_eval)[0]) / 2.
                else:
                    raise ValueError