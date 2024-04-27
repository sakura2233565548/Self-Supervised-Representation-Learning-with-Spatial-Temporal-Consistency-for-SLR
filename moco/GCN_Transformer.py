import os

import torch
from torch.autograd import Variable
from moco.st_gcn_encoder import  st_gcn_single_frame
import torch.nn as nn
import math
import torch.nn.functional as F
from moco.Transformer_backbone import Transformer


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=600):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):
        '''
        :param x: size (B, T, C)
        :return: x+position_encoding(x)
        '''
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Embed(nn.Module):
    def __init__(self, gcn_args, d_model=512, dropout=0.1):
        super(Embed, self).__init__()
        gcn_args.layout_encoder = 'stb'
        self.st_gcn_hand = st_gcn_single_frame.Model(opt=gcn_args)
        gcn_args.layout_encoder = 'body'
        self.st_gcn_body = st_gcn_single_frame.Model(opt=gcn_args)

    def forward(self, pose):
        rh = pose['rh']
        lh = pose['lh']
        body = pose['body']
        rh_feat = self.st_gcn_hand(rh)
        lh_feat = self.st_gcn_hand(lh)
        body_feat = self.st_gcn_body(body)
        hand_feat = torch.cat([rh_feat, lh_feat], dim=2)
        return hand_feat, body_feat


class SingleWord_Classfication_with_dropout_V2(torch.nn.Module):
    def __init__(self, in_channels, num_class, dropout, inter_dist):
        super(SingleWord_Classfication_with_dropout_V2, self).__init__()
        self.fc = torch.nn.Linear(in_channels, num_class)
        self.weight_hand = torch.nn.Linear(in_channels, 1)
        self.dropout = dropout
        self.inter_dist = inter_dist
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.weight_body = torch.nn.Linear(in_channels, 1)
        print('>> Fc Dropout:', dropout)
        if self.inter_dist is True:
            self.body_fc = torch.nn.Linear(in_channels, num_class)
            self.hand_fc = torch.nn.Linear(in_channels, num_class)

    def initialization(self, network):
        print("<<< initialize the para in prediction head!")
        for p in network.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None, seg_embed=None, knn_eval=False):
        """
        :param x: B * 2T * C
        :param mask: B * T * 1
        :return: pred score
        """
        B, T, C = x.size()
        right_hand_feat = x[:, :, 0:512]
        left_hand_feat = x[:, :, 512:1024]
        body_feat = x[:, :, 1024:]
        # x = x[:, 0:T//2, :] + x[:, T//2:, :]

        # Hand feature merge
        hand_feat = torch.cat([right_hand_feat, left_hand_feat], dim=1)
        weight_hand = self.weight_hand(hand_feat)
        if mask is not None:
            weight_hand = weight_hand.masked_fill(mask == 0, -1e9)
        weight_hand = F.softmax(weight_hand, dim=1)
        hand_feat = hand_feat.permute(0, 2, 1)
        hand_feat = torch.matmul(hand_feat, weight_hand).squeeze()

        # Body feature merge
        weight_body = self.weight_body(body_feat)
        weight_body = F.softmax(weight_body, dim=1)
        body_feat = body_feat.permute(0, 2, 1)
        body_feat = torch.matmul(body_feat, weight_body).squeeze()

        feat = body_feat + hand_feat
        if self.dropout is not None:
            feat = self.dropout(feat)
        if knn_eval is True:
            return feat
        else:
            pred = self.fc(feat)
            if self.inter_dist is True:
                return pred, self.hand_fc(hand_feat), self.body_fc(body_feat)
            else:
                return pred, hand_feat, body_feat


class moco_model_with_transfer(torch.nn.Module):
    def __init__(self, opt):
        super(moco_model_with_transfer, self).__init__()
        self.embed = Embed(gcn_args=opt, d_model=opt.hidden_dim_hand, dropout=opt.dropout)
        self.GCN_Tran_Hand = Transformer(dim=opt.hidden_dim_hand, n_heads=opt.heads, dim_ff=opt.d_ff_hand, blocks=opt.blocks,
                                     dropout=opt.dropout)
        self.GCN_Tran_Body = Transformer(dim=opt.hidden_dim_body, n_heads=opt.heads, dim_ff=opt.d_ff_body, blocks=opt.blocks,
                                     dropout=opt.dropout)
        self.proj = SingleWord_Classfication_with_dropout_V2(opt.input_dim, opt.num_class, dropout=opt.proj_dropout, inter_dist=opt.inter_dist)
        self.input_size = opt.input_size

    def forward(self, input, knn_eval=False):
        hand_feat, body_feat = self.embed(input)
        mask = torch.ones((hand_feat.shape[0], hand_feat.shape[1]), device=hand_feat.device)
        hand_feat_refine = self.GCN_Tran_Hand(hand_feat, mask)
        body_feat_refine = self.GCN_Tran_Body(body_feat, mask)
        mask = torch.sum(torch.sum(input['mask'].cuda() > 0, dim=-1), dim=-1, keepdim=True) > 0
        pose_feat_refine = torch.cat([hand_feat_refine, body_feat_refine], dim=2)
        pred_feat, hand_feat, body_feat = self.proj(pose_feat_refine, mask, knn_eval)
        return pred_feat, hand_feat, body_feat


class Config:
    hidden_dim_hand = 1024
    hidden_dim_body = 512
    proj_dropout = 0.0
    heads = 8
    d_ff_hand = 1536
    d_ff_body = 1024
    blocks = 3
    dropout = 0.1
    input_size = 32
    temporal_pad = 0
    in_channels = 2
    out_channels = 3
    strategy = 'spatial'
    layout_encoder = 'stb'
    num_class = 128
    input_dim = 512
    inter_dist = False

