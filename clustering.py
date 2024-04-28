"""
"""

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from math import ceil
from utils import *
from sklearn.mixture import GaussianMixture
from utils import *
from torch_geometric.utils import degree, add_self_loops

class DIST(object):
    def __init__(self, dist_type):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        """cross是计算大量数据点与少量中心点的距离的参数"""
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        """pointA pointB都是矩阵"""
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))
    def euc(self, pointA, pointB, cross):
        """如果用欧式距离算"""
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return torch.norm(pointA-pointB)
        else:
            return torch.cdist(pointA,pointB,p=2)

class GMMClustering(object):
    def __init__(self, num_class,dist_type='cos'):
        self.Dist = DIST(dist_type)
        self.num_class = num_class

    def forward(self, src_centers, emb_t, y_t, target_edge_index, target_edge_attr, smooth, smooth_r):
        self.gmm = GaussianMixture(n_components=self.num_class, n_init=5)
        self.gt = y_t
        model = self.gmm.fit(emb_t.cpu().numpy())
        self.target_pred = model.predict(emb_t.cpu().numpy())
        self.target_pred_prob = torch.FloatTensor(model.predict_proba(emb_t.cpu().numpy()))
        tgt_centers = get_centers_array(emb_t, self.target_pred)
        cluster2label = self.align_centers(src_centers, tgt_centers)
        shuffle_idx = [x[1] for x in sorted({idx:i for i, idx in enumerate(cluster2label)}.items())]
        self.num_nodes = emb_t.size(0)
        labels = []
        for i in range(self.num_nodes):
            labels.append(cluster2label[self.target_pred[i]])
        self.labels = torch.LongTensor(labels)
        # re-arrange
        self.target_pred_prob = self.target_pred_prob[:,shuffle_idx]
        assert torch.argmax(self.target_pred_prob, dim=1).equal(self.labels)
        if smooth:
            # smooth
            self.smooth_labels, p = self.smooth(target_edge_index.cpu(), target_edge_attr.cpu(), smooth_r)
            # summary
            self.samples = {"data":list(range(emb_t.size(0))), "label":self.smooth_labels, "dist2center":p, "gt":self.gt}
        else:
            self.samples = {"data":list(range(emb_t.size(0))), "label":self.labels, "dist2center":self.target_pred_prob, "gt":self.gt}
        return self.samples

    def smooth(self, edge_index, v, smooth_r):
        from torch_sparse import SparseTensor, spmm
        edge_index_sp = SparseTensor(row=edge_index[0], col=edge_index[1], value=v, sparse_sizes=(self.num_nodes,self.num_nodes))
        A_norm = gcn_norm(edge_index=edge_index_sp)
        row, col, v = A_norm.storage._row, A_norm.storage._col, A_norm.storage._value
        p = self.target_pred_prob
        for i in range(20):
            pred_prob_smooth = spmm(index=torch.vstack([row,col]), value=v, m=self.num_nodes, n=self.num_nodes, matrix=p)
            # 0.8 0.2可以跑出好效果
            label = smooth_r*p+(1-smooth_r)*pred_prob_smooth
            p = label
        return torch.argmax(p, dim=1), p


    def align_centers(self, src_center, tgt_center):
        """为目标域的每个簇重新指派一个源域的簇，重新确定这个簇的类别"""
        cost = self.Dist.get_dist(tgt_center, src_center, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind



class Clustering(object):
    def __init__(self, eps, device,max_len=1000, dist_type='cos'):
        self.eps = eps  # 聚类中心变化的下界
        self.device = device
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers  # [[...],[...],...,[...]]
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps  # 停止条件是中心点变化总和很小

    def assign_fake_labels(self, feats):
        """为目标域节点打伪标签的方法"""
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def align_centers(self):
        """为目标域的每个簇重新指派一个源域的簇，重新确定这个簇的类别"""
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, feat,label):
        """原方法是把dataloader中所有batch的数据拿到内存中，变成self.samples
            但是在图上没有mini batch 是full batch
        """
        self.samples['gt'] = label  # N*5
        self.samples['feature'] = feat  # N*M,M是提取后的特征维度
        self.samples['data'] = list(range(feat.size(0)))

    def feature_clustering(self,feat,label):
        centers = None
        self.stop = False

        self.collect_samples(feat,label)
        feature = self.samples['feature']

        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).to(self.device)
        num_samples = feature.size(0)
        # 这里是对全部样本聚类，所以要分批
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_fake_labels(cur_feature)
                labels_onehot = dataUtils.to_onehot(labels, self.num_classes, self.device)
                # count 数每个类别多少样本
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.FloatTensor).to(self.device)
                # 是否归一化计算聚类中心
                reshaped_feature = cur_feature.unsqueeze(0)
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len
            #centers = centers/count.reshape(self.num_classes,-1)
            mask = (count.unsqueeze(1) > 0).type(torch.FloatTensor).to(self.device)
            # 这步的目的是，如果有类别没有节点，该类的中心就用初始化的中心的代替
            centers = mask * centers + (1 - mask) * self.init_centers

        # 到这里可能聚类中心发生了很小的移动但没到下界，所以跳出了循环，但是需要再重新算一次聚类
        # 前面的kmeans没有真正打上伪标签，只是聚类的过程，聚类完成后才打伪标签
        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_fake_labels(cur_feature)

            labels_onehot = dataUtils.to_onehot(cur_labels, self.num_classes,device=self.device)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        #
        cluster2label = self.align_centers()
        # reorder the centers：重新打标签，根据聚簇距离
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, self.init_centers))

        del self.samples['feature']
        self.samples['label'].cpu()
