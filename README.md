# SkipNode: On Alleviating Over-smoothing for Deep Graph Convolutional Networks

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --data cora --model GCN --layer 8 --skip_type degree --sampling_rate 0.9 --dropout 0.3 --lr 0.01 --wd 5e-4 --epochs 500

python train.py --data cora --model GCN --layer 8 --skip_type random --sampling_rate 0.8 --dropout 0.2 --lr 0.01 --wd 5e-4 --epochs 500

python train.py --data citeseer --model GCN --layer 8 --skip_type degree --sampling_rate 0.9 --dropout 0.2 --lr 0.01 --wd 5e-4 --epochs 500

python train.py --data citeseer --model GCN --layer 8 --skip_type random --sampling_rate 0.9 --dropout 0.2 --lr 0.01 --wd 5e-4 --epochs 500

python train.py --data pubmed --model GCN --layer 8 --skip_type degree --sampling_rate 0.7 --dropout 0.3 --lr 0.01 --wd 5e-4 --epochs 500

python train.py --data pubmed --model GCN --layer 8 --skip_type random --sampling_rate 0.2 --dropout 0.2 --lr 0.01 --wd 5e-4 --epochs 500
```