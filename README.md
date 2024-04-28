### official code for KAIS2023 paper : Node classification across networks via category-level domain adaptive network embedding

#### file structure
/data: 1. Unzip the .zip file to get 3 dirs 2. it has files for citation network acmv9, dblpv7 and citationv1, with both raw data and processed graph object (by PyG)

main.py: entry

models.py: model file

dataloader.py: dataloader

cdd.py: compute intra- and inter- domain discrepancy

clustering.py: generate pseudo labels for target domain

utils.py: help functions

#### run

##### config
```python
torch==1.8.1+cu102
torch-cluster==1.5.9
torch-geometric==2.0.2
torch-scatter==2.0.9
torch-sparse==0.6.12
```

We have also tested under other settings, like torch 1.11.0+cu113&pyg2.4.0. The right combination of these two key packages can run the code normally.

##### command

python main.py --src_name {src_name}  --tgt_name {tgt_name}   ({} from: acmv9, dblpv7, citationv1)



Please cite the following paper if you use this code, thanks a lot!

```latex
@article{shi2023node,
  title={Node classification across networks via category-level domain adaptive network embedding},
  author={Shi, Boshen and Wang, Yongqing and Shao, Jiangli and Shen, Huawei and Li, Yangyang and Cheng, Xueqi},
  journal={Knowledge and Information Systems},
  volume={65},
  number={12},
  pages={5479--5502},
  year={2023},
  publisher={Springer}
}
```