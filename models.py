import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, InnerProductDecoder, GATConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import scipy.sparse as sp

class PPMIGCN(nn.Module):
    def __init__(self,input_dim,hidden_dims):
        self.input_dim = input_dim
        super(PPMIGCN, self).__init__()
        self.conv1 = GATConv(self.input_dim, hidden_dims[0], edge_dim=1)
        self.conv2 = GATConv(hidden_dims[0], hidden_dims[1], edge_dim=1)
        self.prelu = nn.PReLU()
    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.ppmi_edge_index, data.ppmi_edge_attr
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index, edge_attr=edge_attr)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index, edge_attr=edge_attr)))
        return feat2



class GCN(nn.Module):
    def __init__(self,input_dim,hidden_dims):
        self.input_dim = input_dim
        super(GCN, self).__init__()
        self.conv1 = GATConv(self.input_dim, hidden_dims[0], edge_dim=1)
        self.conv2 = GATConv(hidden_dims[0], hidden_dims[1], edge_dim=1)
        self.prelu = nn.PReLU()
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index)))
        return feat2

class Attention(nn.Module):
    """
    该种实现方法用在UDAGCN、ASN中，但并没有像论文中那样把原始输入建模进来
    """
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        # [bs,2,128]
        stacked = torch.stack(inputs, dim=1)
        # [bs,2,1]
        weights = F.softmax(self.dense_weight(stacked), dim=1)  # 因为这里是三维张量，所以dim=1实际上就是2维情况的dim=0
        # [bs,128]
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class Attention2(nn.Module):
    """ 实现论文中描述的注意力机制  实验表明f = Q^T*K 比归一化要效果好"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, out_channels)  # change Zo to the shape of Zl、Zg
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        """inputs = [Zo, Zl, Zg]"""
        Z_o = inputs[0]
        Z_l = inputs[1]
        Z_g = inputs[2]
        Z_o_trans = self.dense_weight(Z_o)
        att_l = torch.sum(torch.matmul(Z_o_trans, torch.t(Z_l)),dim=1)
        att_g = torch.sum(torch.matmul(Z_o_trans, torch.t(Z_g)),dim=1)
        att = torch.hstack([att_l.view(-1,1), att_g.view(-1,1)])
        att = F.softmax(att, dim=1)
        outputs = Z_l*(att[:,0].view(-1,1))+Z_g*(att[:,1].view(-1,1))
        return outputs

class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_channels,out_channels)
        std = 1/(in_channels/2)**0.5
        nn.init.trunc_normal_(self.fc.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.fc.bias, 0.1)
    def forward(self, x):
        logits = self.fc(x)
        return logits

class GraphDA(nn.Module):
    def __init__(self, attn_mode, input_dim, output_dim, hidden_dims, in_channels, out_channels=0):
        """

        :param attn_mode: 0:只使用ppmi /1: 简单ppmi-adj注意力 / 2: 复杂ppmi-adj注意力
        :param input_dim:
        :param output_dim:
        :param hidden_dims:
        :param in_channels:
        :param out_channels:
        """
        super(GraphDA, self).__init__( )
        self.gcn = GCN(input_dim, hidden_dims)
        self.ppmi_gcn = PPMIGCN(input_dim, hidden_dims)
        self.attn_mode = attn_mode
        if attn_mode==1:
            self.attn = Attention(in_channels)
        elif attn_mode==2:
            self.attn = Attention2(in_channels,out_channels)
        self.clf = Classifier(in_channels, output_dim)

    def forward(self, data):
        l_out = self.gcn(data)
        g_out = self.ppmi_gcn(data)
        if self.attn_mode==0:
            emb = g_out
        elif self.attn_mode==1:
            emb = self.attn([l_out, g_out])
        else:
            emb = self.attn([data.x, l_out, g_out])
        pred = self.clf(emb)
        return emb, pred


def net_pro_loss(emb, edge_index, edge_value, num_node, device, choice, mode="ppmi"):
    """
    network proximity loss from ACDNE/neg_sampling/bce
    :param emb:
    :param a:
    :param mode:
    :return:
    """
    # 处理空子图
    if edge_index.shape[1]==0: return 0.
    if choice=="ACDNE":
        # edge_index to dense matrix
        if mode=="ppmi":
            a = sp.coo_matrix((edge_value,(edge_index[0,:], edge_index[1,:])),shape=(num_node, num_node)).toarray()
        else:
            a = sp.coo_matrix((np.ones_like(edge_index[0,:]),(edge_index[0,:], edge_index[1,:])),shape=(num_node, num_node)).toarray()
        a = torch.FloatTensor(a).to(device)
        r = torch.sum(emb*emb, 1)
        r = torch.reshape(r, (-1, 1))
        dis = r-2*torch.matmul(emb, emb.T)+r.T
        return torch.mean(torch.sum(a.__mul__(dis), 1))
    elif choice=="BCE":
        pass
    elif choice=="NEG_SAMPLE":
        pos_loss = 0.
        neg_loss = 0.
        decoder = InnerProductDecoder().to(device)
        EPS = 1e-15
        # 下面这行可能在不同的代码里开关不一样
        if type(edge_index)!=torch.Tensor:
            edge_index = torch.LongTensor(edge_index)
        pos_loss = -torch.log(decoder(emb, edge_index.to(device), sigmoid=True) + EPS).mean()
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(edge_index)
        #pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, emb.size(0),num_neg_samples=4*pos_edge_index.shape[1])
        # 负采样可能得到空负边集
        if min(neg_edge_index.shape)!=0:
            neg_loss = -torch.log(1 - decoder(emb, neg_edge_index, sigmoid=True) +EPS).mean()
        return pos_loss + neg_loss
