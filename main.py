import argparse
import os.path

from strategies.strategy import Strategy
from utils.dataset import Graph
from utils.train import run_training
import torch
import torch.nn as nn
from models.gnn import GNN
import random
import numpy as np
from logger import Logger
import json


def seed_setting(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        pass


def generate_strategy(args):
    if args.strategy.lower() == "none":
        args.drop_edge_ratio = 0.
        args.skip_node_ratio = 0.
        args.pair_norm_scale = 0.
        args.drop_message_ratio = 0.
        args.drop_node_ratio = 0.
        args.skip_node_type = "None"

    strategy_param = {
        "name": args.strategy,
        "drop_edge_ratio": args.drop_edge_ratio,
        "skip_node_ratio": args.skip_node_ratio,
        "skip_node_type": args.skip_node_type,
        "pair_norm_scale": args.pair_norm_scale,
        "drop_message_ratio": args.drop_message_ratio,
        "drop_node_ratio": args.drop_node_ratio
    }

    strategy = Strategy(strategy_param)

    return strategy


def get_args():
    parser = argparse.ArgumentParser(description='GNN Training and Evaluation')
    parser.add_argument('--model', type=str, default='GCN', help='Name of the GNN model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--hid_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout on features")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay rate')
    parser.add_argument('--grand_prop_times', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='cora', help='Name of the dataset')
    parser.add_argument('--setting', type=str, default='semi', help='semi or full')

    parser.add_argument('--strategy', type=str, default='None', help='Name of the strategy')
    parser.add_argument('--drop_edge_ratio', type=float, default=0.2)
    parser.add_argument('--drop_node_ratio', type=float, default=0.2)
    parser.add_argument('--skip_node_ratio', type=float, default=0.5)
    parser.add_argument('--drop_message_ratio', type=float, default=0.5)
    parser.add_argument('--skip_node_type', type=str, default="b")
    parser.add_argument('--pair_norm_scale', type=float, default=1)

    # for random graph generation
    parser.add_argument('--random_graph_nodes', type=int, default=2000)
    parser.add_argument('--random_graph_density', type=float, default=100)
    parser.add_argument('--random_graph_class', type=int, default=3)
    parser.add_argument('--random_graph_feat_dim', type=int, default=16)

    parser.add_argument('--use_param', action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    ratio = None
    if args.setting == "full":
        ratio = (0.6, 0.2, 0.2)

    # for setting
    if args.dataset.lower() in ("cornell", "texas", "chameleon", "wisconsin"):
        args.setting = "full"

    if args.use_param:
        if args.setting == "semi":
            json_file_path = f"./param/{args.setting}/{args.strategy}/{args.dataset}_{args.model}_{args.strategy}_{args.num_layers}.json"
            if not os.path.exists(json_file_path):
                json_file_path = f"./param/{args.setting}/None/{args.dataset}_{args.model}_None_{args.num_layers}.json"
        elif args.setting == "full":
            json_file_path = f"./param/{args.setting}/{args.strategy}/{args.dataset}_{args.model}_{args.strategy}.json"
            if not os.path.exists(json_file_path):
                json_file_path = f"./param/{args.setting}/None/{args.dataset}_{args.model}_None.json"
        else:
            raise ValueError(f"Not supported setting {args.setting}")
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # update args
        param_dict = dict(json_data)
        param_dict["runs"] = args.runs
        param_dict["strategy"] = args.strategy
        param_dict["drop_edge_ratio"] = args.drop_edge_ratio
        param_dict["drop_node_ratio"] = args.drop_node_ratio
        param_dict["skip_node_ratio"] = args.skip_node_ratio
        param_dict["drop_message_ratio"] = args.drop_message_ratio
        param_dict["skip_node_type"] = args.skip_node_type
        param_dict["pair_norm_scale"] = args.pair_norm_scale
        param_dict["dropout"] = args.dropout
        args.__dict__.update(param_dict)

    results = []
    logger = Logger(args)
    print(f"All {args.runs} Runs")
    for i in range(args.runs):
        seed = args.seed + i * 10
        seed_setting(seed)
        data = Graph(root='data/', name=args.dataset, device=device, ratio=ratio, seed=seed,
                     num_nodes=args.random_graph_nodes, density=args.random_graph_density,
                     num_classes=args.random_graph_class, feat_dim=args.random_graph_feat_dim)
        if args.dataset == "random":
            logger.name_change(data.name)
        strategy = generate_strategy(args)
        model = GNN(
            in_channels=data.num_features,
            hid_channels=args.hid_channels,
            out_channels=data.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            layer_name=args.model,
            strategy=strategy,
            bias=True
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        criterion = nn.CrossEntropyLoss()

        if args.model != "GRAND":
            args.grand_prop_times = 1
        test_acc, run_time = run_training(model, optimizer, criterion, data, args)
        print(f"RUN {i + 1} Final Acc:{test_acc}")
        results.append(test_acc)
        logger.add_result(i, test_acc, "acc")
        logger.add_result(i, run_time, "time")


    print(f"Final Result :${np.mean(results):.2f}_",
          "{", f"\pm{np.std(results):.2f}", "}$ at ",
          f"{args.dataset} with {args.runs} runs",
          sep="")

    logger.print_statistics()


if __name__ == "__main__":
    main()
