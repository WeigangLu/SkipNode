# SkipNode: On Alleviating Performance Degeneration for Deep Graph Convolutional Networks

## Installation

```
pip install -r reqyirements.txt
```



## Usage

```
python main.py --use_param --strategy {SkipNode, DropEdge, ..., None} --skip_node_type {u,b} --model {GCN, GAT, ...} --dataset {cora, citeseer, pubmed} 
```



## Citation

> ```
> @article{lu2021skipnode,
>   title={SkipNode: On Alleviating Performance Degradation for Deep Graph Convolutional Networks},
>   author={Lu, Weigang and Zhan, Yibing and Lin, Binbin and Guan, Ziyu and Liu, Liu and Yu, Baosheng and Zhao, Wei and Yang, Yaming and Tao, Dacheng},
>   journal={arXiv preprint arXiv:2112.11628},
>   year={2021}
> }
> ```