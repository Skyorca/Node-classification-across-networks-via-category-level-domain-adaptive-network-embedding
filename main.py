"""
使用高斯混合聚类
加入平滑
"""
from dataloader import CitationDomainData
from clustering import  GMMClustering
from cdd import CDD
from  utils import *
from models import *
from argparse import ArgumentParser
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
parser = ArgumentParser()
######## Input ###########
parser.add_argument("--src_name", type=str)
parser.add_argument("--tgt_name", type=str)
parser.add_argument("--cdd_weight",type=float, default=10.)
parser.add_argument("--clf_weight",type=float, default=1.)
parser.add_argument("--min_sn_cls", type=int, default=10)
parser.add_argument("--epoch", type=int, default=20)

args = parser.parse_args()
#with open("ablation_study_7.txt",'a') as f:
#    f.write(args.src_name+","+args.tgt_name+'\n')



src = CitationDomainData(f"data/{args.src_name}",name=args.src_name,use_pca=False)
tgt = CitationDomainData(f"data/{args.tgt_name}",name=args.tgt_name,use_pca=False)
src_data = src[0].to(device)
tgt_data = tgt[0].to(device)
inp_dim = src_data.x.shape[1]
out_dim = src_data.y.shape[1]
num_nodes_s = src_data.x.shape[0]
num_nodes_t = tgt_data.x.shape[0]
num_class = out_dim

######## HyperParams & Switches ###########
attn_mode = 1
hidden_dims = [128,16]
kernel_num = (10,10)
kernel_mul = (2,2)
num_layers = 1
eps = 1e-3
filter_threshold = 1
# 是否要随着样本数增加扩大batch?
min_sn_cls = args.min_sn_cls
# 放大cdd weight是有益的
cdd_weight = args.cdd_weight
clf_weight = args.clf_weight
entropy_loss_weight = 0.5
net_pro_weight = 0.2
batch_size = min_sn_cls*out_dim*2
# lr小一点比较好，调到0.05时震荡太厉害
lr_ini = 0.02
EPOCH = args.epoch
choice = "ACDNE"
smooth = True
smooth_r = 0.8

# models
model = GraphDA(attn_mode,inp_dim,out_dim, hidden_dims,in_channels=hidden_dims[-1]).to(device)
criterion = nn.BCEWithLogitsLoss(reduction='none')  # 这里需要reduction=none，否则就会求所有entry的平均，但我们需要求所有样本的平均

model.zero_grad()
cdd = CDD(kernel_num=kernel_num, kernel_mul=kernel_mul, num_layers=num_layers, num_classes=out_dim, device=device, intra_only=False)
running_loss = 0.

start = time.time()
for epoch in range(EPOCH):
    best_macro = 0.
    best_micro = 0.
    epoch_loss = 0.
    # cdd weight 变化  系数是10的时候最优， 5可能会差点
    cdd_weight += ((epoch+1)/EPOCH)*10
    p = float(epoch) / EPOCH
    lr = lr_ini / (1. + 10*p) ** 0.75
    #dd_weight = max(cdd_weight, 2/(1+np.exp(-1*epoch))-1)
    optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=5e-4)
    max_iter = max(200,int(max(num_nodes_t, num_nodes_s)/(batch_size/2)))
    iter_time = 0
    model.eval()
    # 聚类算法
    with torch.no_grad():
        emb_s,_ = model(src_data)
        emb_t,_ = model(tgt_data)
        src_centers = get_centers(emb_s, src_data.y)
        clustering = GMMClustering(num_class=num_class)
        target_samples = clustering.forward(src_centers, emb_t, tgt_data.y, tgt_data.edge_index,torch.ones(tgt_data.edge_index.size(1)),smooth, smooth_r )
        # 没有过滤，靠smoothing修正
        target_filter_samples, filter_classes = filtering(threshold=filter_threshold, min_sn_cls=min_sn_cls, target_samples=target_samples, num_class=num_class)
    while True:
        model.train()
        # TODO 加入内循环停止条件
        if iter_time>max_iter: break
        iter_time += 1
        # 采样 feat是原始特征 label_s是one-hot  label_t_fake是标签号向量
        s_idx, feat_s, label_s_batch, t_idx, feat_t, label_t_fake_batch = class_aware_sampling(target_filter_samples,filter_classes,src_data.x, src_data.y, tgt_data.x, num_class, min_sn_cls)
        # 前向传播
        # 因为model在每个mini-batch都更新，所以在全图上重新运行
        model.train()
        emb_s, pred_s = model(src_data)
        emb_t, pred_t = model(tgt_data)
        emb_s_batch = emb_s[s_idx]
        pred_s_batch = pred_s[s_idx]
        emb_t_batch = emb_t[t_idx]
        pred_t_batch = pred_t[t_idx]
        # mini-batch 对应子图
        if choice =="ACDNE":
            edges_s, values_s = batch_edges(s_idx,src_data.ppmi_edge_index,src_data.ppmi_edge_attr,"ppmi")
            edges_t, values_t = batch_edges(t_idx, tgt_data.ppmi_edge_index,tgt_data.ppmi_edge_attr,"ppmi")
        else:
            edges_s, values_s = batch_edges(s_idx,src_data.edge_index,0)
            edges_t, values_t = batch_edges(t_idx, tgt_data.edge_index,0)

        # 图重构损失
        network_loss = net_pro_loss(emb_s_batch,edges_s, values_s,int(batch_size/2), device, choice=choice)+net_pro_loss(emb_t_batch, edges_t, values_t, int(batch_size/2), device, choice=choice)
        # 源域分类损失
        clf_loss = torch.sum(criterion(pred_s_batch, label_s_batch))/(int(batch_size/2))
        # entropy loss
        target_probs = F.softmax(pred_t_batch, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
        entropy_loss = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
        # 带置信度的cdd loss
        weight_ss, weight_tt, weight_st = get_weights(label_s_batch, F.sigmoid(pred_t_batch), [min_sn_cls]*num_class, [min_sn_cls]*num_class)
        cdd_loss = {}
        if num_layers ==1:
            cdd_loss = cdd.forward(source=[emb_s_batch], target=[emb_t_batch],nums_S=[min_sn_cls]*num_class, nums_T=[min_sn_cls]*num_class, weight_ss=weight_ss, weight_tt=weight_tt, weight_st=weight_st)
        else:
            # multi-layer cdd 把embedding结果和classifier结果都放进去
            cdd_loss = cdd.forward(source=[emb_s.to(device), pred_s.to(device)], target=[emb_t.to(device), pred_t.to(device)],nums_S=[min_sn_cls]*num_class, nums_T=[min_sn_cls]*num_class,weight_ss=weight_ss, weight_tt=weight_tt, weight_st=weight_st)

        # 总损失
        loss = net_pro_weight*network_loss + clf_weight*clf_loss + entropy_loss_weight*entropy_loss + cdd_weight*cdd_loss['cdd']
        epoch_loss += loss
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate
        if iter_time%40==0:
            model.eval()
            with torch.no_grad():
                _, pred_s = model(src_data)
                _, pred_t = model(tgt_data)
            pred_s = F.sigmoid(pred_s)
            pred_t = F.sigmoid(pred_t)
            f1_s = f1_scores(pred_s.cpu().numpy(), src_data.y.cpu().numpy())
            f1_t = f1_scores(pred_t.cpu().numpy(), tgt_data.y.cpu().numpy())
            print(epoch, iter_time, f1_s, f1_t)
            print(f"total loss:{loss.item()}, clf loss:{clf_loss.item()}, cdd intra:{cdd_loss['intra']}, cdd inter:{cdd_loss['inter']}, entorpy loss:{entropy_loss.item()}")
            best_micro = max(f1_t[0], best_micro)
            best_macro = max(f1_t[1], best_macro)
            line = f"Epoch:{epoch},Iter:{iter_time},MicroF1:{best_micro},MacroF1:{best_macro}\n"
            end = time.time()
            print(end-start)
    model.eval()
    with torch.no_grad():
        _, pred_s = model(src_data)
        _, pred_t = model(tgt_data)
    pred_s = F.sigmoid(pred_s)
    pred_t = F.sigmoid(pred_t)
    f1_s = f1_scores(pred_s.cpu().numpy(), src_data.y.cpu().numpy())
    f1_t = f1_scores(pred_t.cpu().numpy(), tgt_data.y.cpu().numpy())
    end = time.time()
    print(end-start)
    best_micro = max(best_micro, f1_t[0])
    with open(f"f1_our_{args.src_name}_{args.tgt_name}.txt",'a') as h:
        c = f"{best_micro}\n"
        h.write(c)




