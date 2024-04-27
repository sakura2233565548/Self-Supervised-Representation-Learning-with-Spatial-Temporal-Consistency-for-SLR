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
                 teacher_T=0.05, student_T=0.1, kt_weight=1.0, topk=1024, mlp=False, pretrain=True, dropout=None,
                 inter_weight=0.5, inter_dist=False, topk_part=1024, K_part=8192):
        super(MoCo, self).__init__()
        self.pretrain = pretrain
        RHand_Bone = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 1), (7, 6), (8, 7), (9, 8), (10, 1),
                     (11, 10), (12, 11), (13, 12), (14, 1), (15, 14), (16, 15), (17, 16), (18, 1),
                     (19, 18), (20, 19), (21, 20)]
        Body_Bone = [(2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6)]
        self.Bone = RHand_Bone + [(k+21, v+21) for k,v in RHand_Bone] + [(k+42, v+42) for k,v in Body_Bone]
        if not self.pretrain:
            cfg= Config()
            cfg.proj_dropout = dropout
            cfg.num_class = num_class
            self.encoder_q = moco_model_with_transfer(cfg)
            self.encoder_q_motion = moco_model_with_transfer(cfg)
            init_para_GCN_Trans(self.encoder_q)
            init_para_GCN_Trans(self.encoder_q_motion)
        else:
            self.K = K
            self.m = m
            self.T = T
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.kt_weight = kt_weight
            self.inter_weight = inter_weight
            self.topk = topk
            self.K_part = K_part
            self.topk_part = topk_part
            mlp=mlp
            print(" MoCo parameters",K,m,T,mlp)
            print(" kt parameters: teacher-T %.2f, student-T %.2f, kt-weight: %.2f, topk: %d"%(teacher_T,student_T,kt_weight,topk))
            print(skeleton_representation)

            cfg = Config()
            cfg.proj_dropout = dropout
            cfg.inter_dist = inter_dist
            self.encoder_q = moco_model_with_transfer(cfg)
            self.encoder_k = moco_model_with_transfer(cfg)
            self.encoder_q_motion = moco_model_with_transfer(cfg)
            self.encoder_k_motion = moco_model_with_transfer(cfg)
            init_para_GCN_Trans(self.encoder_q)
            init_para_GCN_Trans(self.encoder_q_motion)
            init_para_GCN_Trans(self.encoder_k)
            init_para_GCN_Trans(self.encoder_k_motion)

            #projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.proj.fc.weight.shape[1]
                self.encoder_q.proj.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_q.proj.fc)
                self.encoder_k.proj.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_k.proj.fc)
                self.encoder_q_motion.proj.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                            nn.ReLU(),
                                                            self.encoder_q_motion.proj.fc)
                self.encoder_k_motion.proj.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                            nn.ReLU(),
                                                            self.encoder_k_motion.proj.fc)
                if inter_dist is True:
                    self.encoder_q.proj.hand_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                           nn.ReLU(),
                                                           self.encoder_q.proj.hand_fc)
                    self.encoder_k.proj.hand_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                           nn.ReLU(),
                                                           self.encoder_k.proj.hand_fc)
                    self.encoder_q_motion.proj.hand_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                                  nn.ReLU(),
                                                                  self.encoder_q_motion.proj.hand_fc)
                    self.encoder_k_motion.proj.hand_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                                  nn.ReLU(),
                                                                  self.encoder_k_motion.proj.hand_fc)
                    self.encoder_q.proj.body_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                                nn.ReLU(),
                                                                self.encoder_q.proj.body_fc)
                    self.encoder_k.proj.body_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                                nn.ReLU(),
                                                                self.encoder_k.proj.body_fc)
                    self.encoder_q_motion.proj.body_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                                       nn.ReLU(),
                                                                       self.encoder_q_motion.proj.body_fc)
                    self.encoder_k_motion.proj.body_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                                       nn.ReLU(),
                                                                       self.encoder_k_motion.proj.body_fc)


            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))
            if inter_dist is True:
                self.register_buffer("queue_hand", torch.randn(dim, self.K_part))
                self.queue_hand = F.normalize(self.queue_hand, dim=0)
                self.register_buffer("queue_ptr_hand", torch.zeros(1, dtype=torch.long))

                self.register_buffer("queue_hand_motion", torch.randn(dim, self.K_part))
                self.queue_hand_motion = F.normalize(self.queue_hand_motion, dim=0)
                self.register_buffer("queue_ptr_hand_motion", torch.zeros(1, dtype=torch.long))

                self.register_buffer("queue_body", torch.randn(dim, self.K_part))
                self.queue_body = F.normalize(self.queue_body, dim=0)
                self.register_buffer("queue_ptr_body", torch.zeros(1, dtype=torch.long))

                self.register_buffer("queue_body_motion", torch.randn(dim, self.K_part))
                self.queue_body_motion = F.normalize(self.queue_body_motion, dim=0)
                self.register_buffer("queue_ptr_body_motion", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        self.queue_motion[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr_motion[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_hand(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_hand)
        self.queue_hand[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K_part  # move pointer
        self.queue_ptr_hand[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_hand_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_hand_motion)
        self.queue_hand_motion[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K_part  # move pointer
        self.queue_ptr_hand_motion[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_body(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_body)
        self.queue_body[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K_part  # move pointer
        self.queue_ptr_body[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_body_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_body_motion)
        self.queue_body_motion[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K_part  # move pointer
        self.queue_ptr_body_motion[0] = ptr


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
            else:
                if view == 'joint':
                    return self.encoder_q(im_q, knn_eval)
                elif view == 'motion':
                    return self.encoder_q_motion(im_q_motion, knn_eval)
                elif view == 'all':
                    total_feat, hand_feat, body_feat = self.encoder_q(im_q, knn_eval)
                    total_feat_motion, hand_feat_motion, body_feat_motion = self.encoder_q_motion(im_q_motion, knn_eval)
                    return (total_feat+total_feat_motion) / 2.0, (hand_feat+hand_feat_motion) / 2.0, (body_feat+body_feat_motion) / 2.0
                            
                else:
                    raise ValueError

        im_k_motion = {}
        for k, v in im_k.items():
            if k in ['rh', 'lh', 'body']:
                temp_tensor = torch.zeros_like(v)
                temp_tensor[:, :-1, :, :] = v[:, 1:, :, :] - v[:, :-1, :, :]
                im_k_motion[k] = temp_tensor
            else:
                im_k_motion[k] = v

        # compute query features
        q, q_hand, q_body = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        if self_dist is True:
            q_hand = F.normalize(q_hand, dim=1)
            q_body = F.normalize(q_body, dim=1)

        q_motion, q_motion_hand, q_motion_body = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)
        if self_dist is True:
            q_motion_hand = F.normalize(q_motion_hand, dim=1)
            q_motion_body = F.normalize(q_motion_body, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()

            k, k_hand, k_body = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if self_dist is True:
                k_hand = F.normalize(k_hand, dim=1)
                k_body = F.normalize(k_body, dim=1)

            k_motion, k_motion_hand, k_motion_body = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)
            if self_dist is True:
                k_motion_hand = F.normalize(k_motion_hand, dim=1)
                k_motion_body = F.normalize(k_motion_body, dim=1)

        # MOCO
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # self_dist negative logits: NxK
        if self_dist is True:
            l_hand_pos = torch.einsum('nc,nc->n', [q_hand, k_hand]).unsqueeze(-1)
            l_body_pos = torch.einsum('nc,nc->n', [q_body, k_body]).unsqueeze(-1)
            l_hand_neg = torch.einsum('nc,ck->nk', [q_hand, self.queue_hand.clone().detach()])
            l_body_neg = torch.einsum('nc,ck->nk', [q_body, self.queue_body.clone().detach()])


        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])
        # self_dist negative logits: NxK
        if self_dist is True:
            l_hand_pos_motion = torch.einsum('nc,nc->n', [q_motion_hand, k_motion_hand]).unsqueeze(-1)
            l_body_pos_motion = torch.einsum('nc,nc->n', [q_motion_body, k_motion_body]).unsqueeze(-1)
            l_hand_neg_motion = torch.einsum('nc,ck->nk', [q_motion_hand, self.queue_hand_motion.clone().detach()])
            l_body_neg_motion = torch.einsum('nc,ck->nk', [q_motion_body, self.queue_body_motion.clone().detach()])


        # self dist loss
        loss_inter_dist = 0.0
        logits_hand = 0.0
        logits_body = 0.0
        logits_hand_motion = 0.0
        logits_body_motion = 0.0
        if self_dist is True:
            # logits: Nx(1+K)
            logits_hand = torch.cat([l_hand_pos, l_hand_neg], dim=1)
            logits_body = torch.cat([l_body_pos, l_body_neg], dim=1)
            logits_hand_motion = torch.cat([l_hand_pos_motion, l_hand_neg_motion], dim=1)
            logits_body_motion = torch.cat([l_body_pos_motion, l_body_neg_motion], dim=1)

            # apply temperature
            logits_hand /= self.T
            logits_body /= self.T
            logits_hand_motion /= self.T
            logits_body_motion /= self.T
            
            lk_hand_neg = torch.einsum('nc,ck->nk', [k_hand, self.queue_hand.clone().detach()])
            lk_body_neg = torch.einsum('nc,ck->nk', [k_body, self.queue_body.clone().detach()])
            lk_hand_neg_motion = torch.einsum('nc,ck->nk', [k_motion_hand, self.queue_hand_motion.clone().detach()])
            lk_body_neg_motion = torch.einsum('nc,ck->nk', [k_motion_body, self.queue_body_motion.clone().detach()])
            #
            lk_hand_neg_topk, topk_hand_idx = torch.topk(lk_hand_neg, self.topk_part, dim=-1)
            lk_body_neg_topk, topk_body_idx = torch.topk(lk_body_neg, self.topk_part, dim=-1)
            lk_hand_neg_motion_topk, motion_topk_hand_idx = torch.topk(lk_hand_neg_motion, self.topk_part, dim=-1)
            lk_body_neg_motion_topk, motion_topk_body_idx = torch.topk(lk_body_neg_motion, self.topk_part, dim=-1)
            #
            loss_inter_dist = loss_kld(torch.gather(l_body_neg, -1, topk_hand_idx) / self.student_T, lk_hand_neg_topk / self.teacher_T) + \
                              loss_kld(torch.gather(l_hand_neg, -1, topk_body_idx) / self.student_T, lk_body_neg_topk / self.teacher_T) + \
                              loss_kld(torch.gather(l_body_neg_motion, -1, motion_topk_hand_idx) / self.student_T, lk_hand_neg_motion_topk / self.teacher_T) + \
                              loss_kld(torch.gather(l_hand_neg_motion, -1, motion_topk_body_idx) / self.student_T, lk_body_neg_motion_topk / self.teacher_T)

        
        # kt loss
        lk_neg = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        lk_neg_motion = torch.einsum('nc,ck->nk', [k_motion, self.queue_motion.clone().detach()])

        # Top-k
        lk_neg_topk, topk_idx = torch.topk(lk_neg, self.topk, dim=-1)
        lk_neg_motion_topk, motion_topk_idx = torch.topk(lk_neg_motion, self.topk, dim=-1)

        loss_kt = loss_kld(torch.gather(l_neg_motion, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + \
                   loss_kld(torch.gather(l_neg, -1, motion_topk_idx) / self.student_T, lk_neg_motion_topk / self.teacher_T)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        if self_dist is True:
            self._dequeue_and_enqueue_hand(k_hand)
            self._dequeue_and_enqueue_hand_motion(k_motion_hand)
            self._dequeue_and_enqueue_body(k_body)
            self._dequeue_and_enqueue_body_motion(k_motion_body)

        return logits, logits_motion, labels, loss_kt * self.kt_weight, logits_hand, logits_body, logits_hand_motion, logits_body_motion, loss_inter_dist