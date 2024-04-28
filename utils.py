import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sklearn.decomposition import PCA
from warnings import filterwarnings
filterwarnings('ignore')
import networkx as nx
import copy
import torch
from torch.nn import functional as F
from scipy.sparse import lil_matrix


def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y

def load_noisy_networks(file):
    """source target是由同一个网络采样而来"""
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    a1,a2 = disturb_network(a)
    x2 = copy.deepcopy(x)
    y2 = copy.deepcopy(y)
    return a1, x, y, a2, x2, y2



def disturb_network(adj):
    g = nx.from_scipy_sparse_matrix(adj)
    edges_1 = []
    edges_2 = []
    edges = g.edges
    g_1 = nx.Graph()
    g_2 = nx.Graph()
    g_1.add_nodes_from(g.nodes)
    g_2.add_nodes_from(g.nodes)
    alpha_s = 0.8
    alpha_c = 0.
    for edge in edges:
        p = np.random.uniform(0,1)
        if p>1-2*alpha_s+alpha_s*alpha_c and p<=1-alpha_s: edges_1.append(edge)
        elif p>1-alpha_s and p<=1-alpha_s*alpha_c: edges_2.append(edge)
        elif p>1-alpha_s*alpha_c:
            edges_1.append(edge)
            edges_2.append(edge)
        else: continue
    g_1.add_edges_from(edges_1)
    g_2.add_edges_from(edges_2)
    adj_1 = nx.to_scipy_sparse_matrix(g_1, format='csc')  # csc
    adj_2 = nx.to_scipy_sparse_matrix(g_2, format='csc')
    return adj_1, adj_2

def graph_sim(adj1, adj2):
    g1_ = nx.from_scipy_sparse_matrix(adj1)
    g1_label = {pair[0]:pair[1] for pair in g1_.degree()}
    g2_ = nx.from_scipy_sparse_matrix(adj2)
    g2_label = {pair[0]:pair[1] for pair in g2_.degree()}
    inp1 = Graph(adj1,node_labels=g1_label)
    inp2 = Graph(adj2,node_labels=g2_label)
    gk = WeisfeilerLehman(n_iter=10, base_graph_kernel=VertexHistogram, normalize=True)
    sim = gk.fit_transform([inp1, inp2])
    return sim[0,1]

def flip_adj(adj:sp.spmatrix):
    """对邻接矩阵随机删边和加边"""
    adj =  adj.todense()
    sparse = np.sum(adj)/(adj.shape[0]**2)
    drop = 0.9  # 越大，删除的边越多
    M_drop = np.random.uniform(size=(adj.shape[0], adj.shape[1]))
    M_drop = (M_drop>drop).astype(int)
    adj_drop = np.multiply(M_drop,adj)
    print(f'delete: {np.sum(np.abs(adj-adj_drop))}')
    # 元素为1的位置为被删过的边，不允许再加边
    dropped = np.abs(adj-adj_drop)
    # 这些位置变成0
    dropped_stay = np.ones_like(dropped)-dropped
    add = 10*sparse  # 越大，添加的边越多
    M_add = np.random.uniform(size=(adj.shape[0], adj.shape[1]))
    M_add = (M_add<add).astype(int)
    # 不允许被删过的边重新添加回来
    M_add = np.multiply(M_add, dropped_stay)
    adj_mod = M_add + adj_drop
    adj_mod[adj_mod==2]=1
    print(f'add: {np.sum(np.abs(adj_drop-adj_mod))}')
    print(f'total modify: {np.sum(np.abs(adj-adj_mod))}')
    adj_ = sp.csc_matrix(adj_drop)
    return adj_

def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat


def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w


def agg_tran_prob_mat(g, step):
    """aggregated K-step transition probality"""
    g = my_scale_sim_mat(g)
    g = csc_matrix.toarray(g)
    a_k = g
    a = g
    for k in np.arange(2, step+1):
        a_k = np.matmul(a_k, g)
        a = a+a_k/k
    return a


def compute_ppmi(a):
    """compute PPMI, given aggregated K-step transition probality matrix as input"""
    np.fill_diagonal(a, 0)
    a = my_scale_sim_mat(a)   # D^-1 *F
    (p, q) = np.shape(a)
    col = np.sum(a, axis=0)
    col[col == 0] = 1
    ppmi = np.log((float(p)*a)/col[None, :])   # D^-1*F*D^-1(分母就是列归一化)*p(sum_i sum_j Fij)
    idx_nan = np.isnan(ppmi)
    ppmi[idx_nan] = 0
    ppmi[ppmi < 0] = 0
    return ppmi


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    shuffle_index = None
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]


def batch_ppmi(batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
    """return the PPMI matrix between nodes in each batch"""
    # #proximity matrix between source network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
    # #proximity matrix between target network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
    return my_scale_sim_mat(a_s), my_scale_sim_mat(a_t)

def batch_ppmi_single(batch_size, shuffle_index_s, ppmi_s):
    a_s = np.zeros((batch_size, batch_size))
    for ii in range(batch_size ):
        for jj in range(batch_size):
            if ii != jj:
                a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
    return my_scale_sim_mat(a_s)

def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)
    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]

def get_centers(embeddings, labels):
    """CV里面的输入（图片）通常在一开始就归一化了，所以求聚类中心时需要归一化
       embeddings: M*D
       labels:     M*5 multi-class one-hot matrix
       output:     5*D
    """
    #embeddings = F.normalize(embeddings,dim=1)
    count = torch.sum(labels,dim=0).view(labels.size(1),-1)
    return torch.matmul(labels.T,embeddings)

def get_centers_array(embeddings, label_array):
    labels = to_onehot(torch.IntTensor(label_array), 5).cuda()
    return torch.matmul(labels.T,embeddings)

def filter_samples(samples, threshold):
    batch_size_full = len(samples['data'])
    min_dist = torch.min(samples['dist2center'], dim=1)[0]
    mask = min_dist < threshold
    filtered_data = [samples['data'][m] for m in range(mask.size(0)) if mask[m].item() == 1]
    filtered_label = torch.masked_select(samples['label'], mask)
    filtered_gt = samples['gt'][filtered_data] if samples['gt'] is not None else None

    filtered_samples = {}
    filtered_samples['data'] = filtered_data
    filtered_samples['label'] = filtered_label
    filtered_samples['gt'] = filtered_gt

    assert len(filtered_samples['data']) == filtered_samples['label'].size(0)
    print('select %f' % (1.0 * len(filtered_data) / batch_size_full))

    return filtered_samples

def filter_class(labels, num_min, num_classes):
    filted_classes = []
    for c in range(num_classes):
        mask = (labels == c)
        count = torch.sum(mask).item()
        if count >= num_min:
            filted_classes.append(c)

    return filted_classes


def filtering(threshold, min_sn_cls, target_samples, num_class):
    """originally threshold=0.05"""
    # filtering the samples
    chosen_samples = filter_samples(target_samples, threshold=threshold)
    # filtering the classes
    # TODO: 当聚类得到的标签类别只有小于标签类别种类时，我实际上并没有做类别过滤
    filtered_classes = filter_class(chosen_samples['label'], min_sn_cls, num_class)
    print('The number of filtered classes: %d.' % len(filtered_classes))
    return chosen_samples, filtered_classes


def label_sample_mapping(y_s,target_filter_samples,filter_class):
    """
    计算source  target的标签到样本集合的映射，为后续采样准备
    :param y_s: multi-class one-hot matrix
    :param y_t_fake: 1-D tensor
    filter_class: []
    :return: {0:[],1:[],...}
    """
    source_mapping = {}
    target_mapping = {}
    y_t_fake = target_filter_samples['label'].cpu().numpy().tolist()
    y_idx = target_filter_samples['data']
    assert len(y_t_fake)==len(y_idx)
    for i in filter_class:
        source_mapping[i] = torch.nonzero(y_s[:,i])
        source_mapping[i] = source_mapping[i].view(len(source_mapping[i]),).cpu().numpy().tolist()
    for j,l in enumerate(y_t_fake):
        if l not in filter_class: continue
        if l not in target_mapping:
            target_mapping[l] = [y_idx[j]]
        else:
            target_mapping[l].append(y_idx[j])
    return source_mapping, target_mapping



def class_aware_sampling(target_filter_samples, filter_class, x_s, y_s,x_t,num_class,bs_per_class,):
    """
    class aware sampling,这里都是numpy运算没有tensor
    :param target_filter_samples:
    :param filter_class:
    :param x_s/x_t: dense np array
    :param y_s:
    :param num_class:
    :param bs_per_class:
    :return: x_s,y_s,idx_s,x_t,y_t_fake,idx_t
    """
    source_mapping, target_mapping = label_sample_mapping(y_s, target_filter_samples, filter_class)
    s_idx = []
    t_idx = []
    feat_s = []
    feat_t = []
    label_s = []
    label_t = []
    y_t_fake = target_filter_samples['label'].cpu().numpy()
    for l in filter_class:
        # 对每个类别随机采样
        s_rdm = np.random.randint(0,len(source_mapping[l]),bs_per_class)
        s_idx_l = np.array(source_mapping[l])[s_rdm].tolist()  # 真实的样本编号
        s_idx += s_idx_l
        feat_s.append(x_s[s_idx_l])
        label_s.append(y_s[s_idx_l])
        t_rdm = np.random.randint(0,len(target_mapping[l]),bs_per_class)
        t_idx_l = np.array(target_mapping[l])[t_rdm].tolist()  # 真实的样本编号
        t_idx += t_idx_l
        feat_t.append(x_t[t_idx_l])
        label_t.append(np.array([l]*bs_per_class).reshape(bs_per_class,-1))
        # 样本index放进idx列表中
    feat_s = torch.vstack(feat_s)
    feat_t = torch.vstack(feat_t)
    label_s = torch.vstack(label_s)
    label_t = torch.FloatTensor(np.vstack(label_t))
    del x_s
    del x_t
    return s_idx, feat_s, label_s, t_idx, feat_t, label_t


def batch_edges(sample_idx, edge_index, edge_value,mode="adj"):
    """
    对每个mini-batch，从邻接矩阵/ppmi矩阵把相应的子图取出来
    sample_idx:
    :param edge_index: COO-like tensor
    :param edge_value: 0：邻接矩阵 / tensor: ppmi矩阵
    :return: np array 节点的编号从0开始，顺序和传入的sample_idx一一对应
    """
    edge_index = edge_index.cpu().numpy()
    idx_map = {x:i for i, x in enumerate(sample_idx)}
    sample_idx = np.array(sample_idx)
    choose = np.isin(edge_index, sample_idx)
    mask = np.logical_and(choose[0,:], choose[1,:])
    edges = edge_index[:,mask]
    def _map(x):
        return idx_map[x]
    _map = np.vectorize(_map)
    if edges.shape[1]>0:
        edges = _map(edges)
    if mode=="ppmi":
        edge_value = edge_value.cpu().numpy()
        values = edge_value[mask]
        assert edges.shape[1]==values.shape[0]
        return edges, values
    else:
        return edges,0


def to_onehot(label, num_classes):
    identity = torch.eye(num_classes)
    onehot = torch.index_select(identity, 0, label)
    return onehot


def wrong_analyze(y_pred_clf, y_true):
    """

    :param y_pred_clf: n*5 matrix
    :param y_true:   one-hot multi-class matrix n*5
    :param y_fake:  list of labels [...]
    :return:
    """
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        wrong_record = {}   # {"":""}
        fake_num = 0
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1],dtype=np.int)
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            if not (pred_i == y_tru[i,:]).all():
                gt = str(list(np.nonzero(y_tru[i,:])[0]))
                pred = str(list(np.nonzero(pred_i)[0]))
                key = gt+"to"+pred
                if key not in wrong_record: wrong_record[key] = 1
                else: wrong_record[key] += 1
        return wrong_record

    return predict(y_true,y_pred_clf)


def cheat_f1_score_single_class(y_true, y_fake):
    """只用伪标签作为预测结果，变成多标签单分类"""
    y_true = to_onehot(torch.tensor(np.argmax(y_true,1)),y_true.shape[1])
    y_fake = to_onehot(y_fake.cpu(),y_fake.shape[1])
    averages = ["micro", "macro"]
    results = {}
    for average in averages:
        results[average] = f1_score(y_true, y_fake, average=average)
    return results["micro"], results["macro"]


def f1_scores_mix(y_pred, y_true, y_fake):
    """a mixture of clf and fake labels"""
    y_fake = to_onehot(y_fake.cpu(),y_true.shape[1]).numpy()
    a = 1
    b = 0
    mi_s = []
    ma_s = []
    for i in range(1,11):
        a -= i*0.1
        b += i*0.1
        y_pred = a*y_pred+b*y_fake
        mi,ma = f1_scores(y_pred, y_true)
        mi_s.append(mi)
        ma_s.append(ma)
    mi_max = max(mi_s)
    idx = mi_s.index(mi_max)
    return mi_max, ma_s[idx], (1-(idx+1)*0.1, 0+(idx+1)*0.1)

def fake_f1_scores(y_fake, y_true):
    y_fake = to_onehot(y_fake.cpu(),y_true.shape[1]).numpy()
    mi,ma = f1_scores(y_fake, y_true)
    return mi, ma

def get_weights(pred_s, pred_t, num_s,num_t):
    """
    :param pred_s: one-hot gt
    :param pred_t: sigmoid/softmax after pred logits
    :param num_s:
    :param num_t:
    :return:
    weight_ss : (nc,bs,bs) 对应5,8,8即列表装载的5个8*8 同域、同类别样本的权重乘积,list of tensor
    weight_tt: 同上
    weight_st: (nc*bs, nc*bs) 对应 （40,40） 跨域，所有类别的样本权重乘积一起计算。
    这样设计是根据cdd中的计算方式决定的。compute_kernel_dist函数对dist/gamma的处理方式
    """
    weight_ss = []
    weight_tt = []
    vec_s = []
    vec_t = []
    nc = len(num_s)
    start = end = 0
    for i in range(nc):
        start = end
        end = start + num_s[i]
        w = pred_s[start:end,i].reshape(num_s[i],-1)
        # sum()会导致除0
        w_norm = w/w.sum()
        weight_ss.append(w_norm@w_norm.T)
        vec_s.append(w_norm)
    start = end = 0
    for i in range(nc):
        start = end
        end = start + num_t[i]
        w = pred_t[start:end,i].reshape(num_t[i],-1)
        w_norm = w/w.sum()
        weight_tt.append(w_norm@w_norm.T)
        vec_t.append(w_norm)
    vec_s = torch.vstack(vec_s)
    vec_t = torch.vstack(vec_t)
    weight_st = torch.matmul(vec_s, vec_t.T)
    return weight_ss, weight_tt, weight_st


def get_weights_correct(pred_s, pred_t, num_s, num_t, filter_classes:list):
    """
    :param pred_s: one-hot gt
    :param pred_t: sigmoid/softmax after pred logits
    :param num_s: [x,x,0,x,x] 0表示对应的类别2被过滤掉了
    :param num_t:
    filter_classes: list, [0,1,3,4] ...
    :return:
    weight_ss : (nc,bs,bs) 对应5,8,8即列表装载的5个8*8 同域、同类别样本的权重乘积,list of tensor
    weight_tt: 同上
    weight_st: (nc*bs, nc*bs) 对应 （40,40） 跨域，所有类别的样本权重乘积一起计算。
    这样设计是根据cdd中的计算方式决定的。compute_kernel_dist函数对dist/gamma的处理方式
    """
    weight_ss = []
    weight_tt = []
    vec_s = []
    vec_t = []
    nc = len(num_s)
    start = end = 0
    for i in filter_classes:
        start = end
        end = start + num_s[i]
        w = pred_s[start:end,i].reshape(num_s[i],-1)
        # sum()会导致除0
        w_norm = w/w.sum()
        weight_ss.append(w_norm@w_norm.T)
        vec_s.append(w_norm)
    start = end = 0
    for i in filter_classes:
        start = end
        end = start + num_t[i]
        w = pred_t[start:end,i].reshape(num_t[i],-1)
        w_norm = w/w.sum()
        weight_tt.append(w_norm@w_norm.T)
        vec_t.append(w_norm)
    vec_s = torch.vstack(vec_s)
    vec_t = torch.vstack(vec_t)
    weight_st = torch.matmul(vec_s, vec_t.T)
    return weight_ss, weight_tt, weight_st

def get_weights_filtered(pred_s, pred_t, num_s,num_t,filtered_class):
    """
    :param pred_s:
    :param pred_t:
    :param num_s:
    :param num_t:
    :return:
    weight_ss : (nc,bs,bs) 对应5,8,8即列表装载的5个8*8 同域、同类别样本的权重乘积,list of tensor
    weight_tt: 同上
    weight_st: (nc*bs, nc*bs) 对应 （40,40） 跨域，所有类别的样本权重乘积一起计算。
    这样设计是根据cdd中的计算方式决定的。compute_kernel_dist函数对dist/gamma的处理方式
    """
    weight_ss = []
    weight_tt = []
    vec_s = []
    vec_t = []
    nc = len(num_s)
    start = end = 0
    # 如果没有类别被过滤，这项是正确的，但如果有过滤的就不能用这种循环
    for i,c in enumerate(filtered_class):
        start = end
        end = start + num_s[i]
        w = pred_s[start:end,c].reshape(num_s[i],-1)
        w_norm = w/w.sum()
        weight_ss.append(w_norm@w_norm.T)
        vec_s.append(w_norm)
    #softmax or sigmoid
    pred_t = F.sigmoid(pred_t)
    start = end = 0
    for i,c in enumerate(filtered_class):
        start = end
        end = start + num_t[i]
        w = pred_t[start:end,c].reshape(num_t[i],-1)
        w_norm = w/w.sum()
        weight_tt.append(w_norm@w_norm.T)
        vec_t.append(w_norm)
    vec_s = torch.vstack(vec_s)
    vec_t = torch.vstack(vec_t)
    weight_st = torch.matmul(vec_s, vec_t.T)
    return weight_ss, weight_tt, weight_st


def KLDiv(y_pred, y_true):
    y_pred = torch.softmax(y_pred,dim=1)
    y_true = y_true+1e-5
    return y_true*(torch.log(y_true)-torch.log(y_pred))



def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    """
    计算 D^{-1/2}AD^{-1/2}
    :param edge_index:
    :param edge_weight:
    :param num_nodes:
    :param improved:
    :param add_self_loops:
    :param dtype:
    :return:
    """
    from torch_scatter import scatter_add
    from torch_sparse import SparseTensor, fill_diag, matmul, mul
    from torch_sparse import sum as sparsesum
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_geometric.utils import add_remaining_self_loops

    fill_value = 2. if improved else 1.
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def stack_samples(x,y):
    '''把X Y按照类别依次堆叠起来'''
    x_ = []
    y_ = []
    cnt = []
    for i in range(2):
        mask = y[:,i]==1.
        x_.append(x[mask])
        y_.append(y[mask])
        cnt.append(torch.sum(mask).cpu().item())
    x = torch.vstack(x_)
    y = torch.vstack(y_)
    return x,y,cnt