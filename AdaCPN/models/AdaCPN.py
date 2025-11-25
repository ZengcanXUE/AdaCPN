import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
import numpy as np
from tqdm import tqdm
from .BaseModel import BaseModel
import logging
from torch.nn import ZeroPad2d
class AdaCPN(BaseModel):
    def __init__(self, config):
        super(AdaCPN, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.multi_convolution_type = kwargs.get('multi_convolution_type')
        self.multi_output = kwargs.get('multi_output')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.kernel_size1 = kwargs.get('conv_kernel_size1')
        self.kernel_size2 = kwargs.get('conv_kernel_size2')
        self.kernel_type = kwargs.get('conv_kernel_type')
        self.stride = kwargs.get('stride')
        self.emb_dim = {
            'entity': kwargs.get('emb_dim'),
            'relation': self.conv_out_channels * self.kernel_size[0] * self.kernel_size[1],
            'relation1': self.conv_out_channels * 1 * 9,
            'relation2': self.conv_out_channels * 1 * 5, 
        }
        self.reshape[0] =  int(self.emb_dim['entity'] / self.reshape[1])
        assert self.emb_dim['entity'] == self.reshape[0] * self.reshape[1]
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim['entity'])
        self.R = torch.nn.Embedding(self.relation_cnt, self.emb_dim['relation'])
        self.R1 = torch.nn.Embedding(self.relation_cnt, self.emb_dim['relation1'])
        self.R2 = torch.nn.Embedding(self.relation_cnt, self.emb_dim['relation2'])
        
        self.filter1_dim = self.conv_out_channels * self.kernel_size[0] * self.kernel_size[1]
        self.fc1 = torch.nn.Linear(self.filter1_dim, self.emb_dim['entity'])
        
        self.q_size = kwargs.get('q_size')
        self.q_size[0] = self.emb_dim['entity'] 
        self.k_size = kwargs.get('k_size') 
        self.k_size[0] = self.emb_dim['relation']
        self.v_size = kwargs.get('v_size')
        self.v_size[0] = self.emb_dim['relation']
        self.v_size[1] = self.conv_out_channels 
        
        self.Q = nn.Parameter(torch.randn(self.q_size)).to(self.device)
        self.K = nn.Parameter(torch.randn(self.k_size)).to(self.device)
        self.V = nn.Parameter(torch.randn(self.v_size)).to(self.device)

        self.a = kwargs.get('a')
        self.P = torch.zeros(config.get('entity_cnt'), config.get('relation_cnt')).to(self.device)
        self.attention(config.get('data'))

        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout')).to(self.device)
        self.feature_map_drop = torch.nn.Dropout2d(kwargs.get('feature_map_dropout')).to(self.device)
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))
        self.bn0 = torch.nn.BatchNorm2d(1)  
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels).to(self.device)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim['entity']).to(self.device)
        self.bn00 = torch.nn.BatchNorm1d(self.emb_dim['relation']).to(self.device)
        self.bn11 = torch.nn.BatchNorm1d(self.emb_dim['relation1']).to(self.device)
        self.bn22 = torch.nn.BatchNorm1d(self.emb_dim['relation2']).to(self.device)
        self.catbn = torch.nn.BatchNorm2d(self.conv_out_channels).to(self.device)

        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.filtered = [self.reshape[0],self.reshape[1]]

        fc_length = self.filtered[0] * self.filtered[1]
        self.fc = torch.nn.Linear(fc_length, self.emb_dim['entity']).to(self.device)
        self.loss = AdaCPNLoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)
        self.init()


        self.perm = 1
        self.embed_dim = self.emb_dim['entity'] 

        self.k_h = self.reshape[0]
        self.k_w = self.reshape[1]
        
        

        self.multi_convolution  = kwargs.get('multi_convolution')
        self.multi_convolution[0] = self.reshape[0]
        self.multi_convolution[1] = self.reshape[1]
        lpad = (self.multi_convolution[0]-self.reshape[0]+self.kernel_size[0]-1)//2
        rpad = (self.multi_convolution[1]-self.reshape[1]+self.kernel_size[1]-1)//2
        lpad2 = (self.multi_convolution[0]-self.reshape[0]+self.kernel_size1[0]-1)//2
        rpad2 = (self.multi_convolution[1]-self.reshape[1]+self.kernel_size1[1]-1)//2
        lpad3 = (self.multi_convolution[0]-self.reshape[0]+self.kernel_size2[0]-1)//2
        rpad3 = (self.multi_convolution[1]-self.reshape[1]+self.kernel_size2[1]-1)//2
        
        if self.kernel_size[0]%2 == 0:
            self.pad = (lpad,lpad+1,rpad,rpad+1)

        else:
            self.pad = (lpad,rpad)
        
        self.pad1 = (lpad2,rpad2)
        self.pad2= (lpad3,rpad3)

        
        self.fc2 = torch.nn.Linear(self.conv_out_channels*3, self.conv_out_channels)
        if self.kernel_type == "55":
            self.fc3 = nn.Conv2d(3 * self.conv_out_channels, self.conv_out_channels, kernel_size=5, stride=1, padding=2)

        
        if self.multi_output == "Coarse":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

    
    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def attention(self, data):
        for d in tqdm(range(len(data))):
            self.P[data[d][0]][data[d][2]] = self.P[data[d][0]][data[d][2]] + 1
        for i in tqdm(range(self.P.size(0))):
            for j in range(self.P.size(1)):
                self.P[i][j] = torch.log(self.P[i][j] + 1)

    def get_cross(self):
        ent_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0: 
                            temp.append(ent_perm[k, ent_idx]) 
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                        else: 
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
            comb_idx.append(temp)
        cross_patt = torch.LongTensor(np.int32(comb_idx))
        return cross_patt

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        batch_size = batch_h.size(0) 

        E = self.E(torch.tensor(batch_h))
        R = self.R(torch.tensor(batch_r))
        
        R1 = self.fc1(R)

        Q = torch.mm(E, self.Q)
        K = torch.mm(R, self.K)
        V = torch.mm(R, self.V)
        
        res = torch.mm(Q, K.T) / sqrt(self.q_size[1]) + self.a * torch.tensor\
                                (self.P[batch_h][:, batch_r])
        res = torch.softmax(res, dim=1)
        atten = torch.mm(res, V)

        if self.multi_convolution_type == "stack":
            entity = self.E(batch_h).reshape(-1, 1, self.emb_dim['entity'])
            relation = R1.reshape(-1, 1, self.emb_dim['entity'])
            e1 = torch.cat([entity, relation], 1).view(-1, 1, *self.reshape) 


        e1 = self.bn0(e1).view(1, -1, *self.reshape) 
        e1 = self.input_drop(e1)

        r = self.R(batch_r)
    
        r1 = self.R1(batch_r)
        r2 = self.R2(batch_r)
        

        if inverse==True:
            r = -r
        r = self.bn00(r) 
        r = self.input_drop(r)
        r = r.view(-1, 1, *self.kernel_size)
        if self.kernel_size[0]%2 == 0:
            pad = ZeroPad2d(self.pad)
            e11 = pad(e1)
            x1 = F.conv2d(e11, r, groups=batch_size*2, padding=0)
        else:
            x1 = F.conv2d(e1, r, groups=batch_size*2, padding=self.pad)
        x1 = x1.view(batch_size, self.conv_out_channels, *self.filtered)
        x1 = self.bn1(x1)

        if inverse==True:
            r1 = -r1
        r1 = self.bn11(r1) 
        r1 = self.input_drop(r1)
        r1 = r1.view(-1, 1, 1, 9)
        x2 = F.conv2d(e1, r1, groups=batch_size*2, padding=self.pad1)
        x2 = x2.view(batch_size, self.conv_out_channels, *self.filtered) 
        x2 = self.bn1(x2)

        if inverse==True:
            r2 = -r2
        r2 = self.bn22(r2) 
        r2 = self.input_drop(r2)
        r2 = r2.view(-1, 1, 1, 5)
        x3 = F.conv2d(e1, r2, groups=batch_size*2, padding=self.pad2)
        x3 = x3.view(batch_size, self.conv_out_channels, *self.filtered) 
        x3 = self.bn1(x3)

        
        if self.multi_output == "Coarse":
            x = torch.cat([x1, x2, x3], dim=1)
            y = self.avg_pool(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            x = x * y.expand_as(x)
            x = self.fc3(x)
        

        x = F.relu(x)
        x = self.feature_map_drop(x)

        x = atten.view(batch_size, self.conv_out_channels, 1, 1) * x
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        y = torch.sigmoid(x)
        return self.loss(y, batch_t), y

class AdaCPNLoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt):
        super().__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt
    
    def forward(self, batch_p, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            batch_e = torch.zeros(batch_size, self.entity_cnt).to(self.device).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss =  self.loss(batch_p, batch_e) / batch_size
        return loss