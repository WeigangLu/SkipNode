from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
import os
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


class Graph(object):
    def __init__(self, name, root, device, ratio=None, seed=None, num_nodes=None, density=None, num_classes=None, feat_dim=None):

        self.device = device

        self.dataset = get_dataset(name, root, seed, num_nodes, density, num_classes, feat_dim)
        self.data = self.dataset[0]

        self.train_mask = None
        self.test_mask = None
        self.val_mask = None

        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.num_nodes = self.data.num_nodes
        self.y = self.data.y.to(device)
        self.x = self.data.x.to(device)
        self.edge_index = self.data.edge_index.to(device)

        self.name = self.dataset.name

        self.generate_mask(ratio)

    def generate_mask(self, ratio=None):
        # if self.name in ("cora", "citeseer", "pubmed"):
        if ratio is None:
            self.train_mask = self.data.train_mask
            self.test_mask = self.data.test_mask
            self.val_mask = self.data.val_mask
        else:
            self.train_mask, self.test_mask, self.val_mask = self.random_splits(ratio)

    def random_splits(self, ratio=(0.6, 0.2, 0.2)):
        labels = torch.clone(self.y)
        train_ratio, val_ratio, test_ratio = ratio
        assert sum(ratio) == 1.0

        indices = []
        for i in range(self.num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append([index, len(index)])

        train_index = torch.cat([i[:int(train_ratio * num)] for i, num in indices], dim=0)
        val_index = torch.cat([i[int(train_ratio * num): int((train_ratio + val_ratio) * num)] for i, num in indices],
                              dim=0)
        test_index = torch.cat([i[int((train_ratio + val_ratio) * num):] for i, num in indices], dim=0)

        train_mask, val_mask, test_mask = (index_to_mask(train_index, size=self.num_nodes),
                                           index_to_mask(val_index, size=self.num_nodes),
                                           index_to_mask(test_index, size=self.num_nodes))

        return train_mask, val_mask, test_mask


class RandomGraphDataset(InMemoryDataset):

    def __init__(self, root, name, num_nodes, density, num_classes, feat_dim):
        self.name = name
        self.num_nodes = num_nodes
        self.density = density
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        root = osp.join(root, name)
        super(RandomGraphDataset, self).__init__(root, transform=T.NormalizeFeatures(), pre_transform=None)
        data, slices = torch.load(self.processed_paths[0])
        data.train_mask = torch.ones(size=(self.num_nodes,)).bool()
        data.val_mask = torch.zeros(size=(self.num_nodes,)).bool()
        data.test_mask = torch.zeros(size=(self.num_nodes,)).bool()

        self.data, self.slices = self.collate([data])

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['random_graph_data.pt']

    def download(self):
        # Downloading is not required
        pass

    def process(self):
        processed_path = self.processed_paths[0]
        if not os.path.exists(processed_path):
            data = generate_random_graph(self.num_nodes, self.density, self.num_classes, self.feat_dim)
            data = self.transform(data)
            torch.save((data, data.contiguous()), processed_path)

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

    def __repr__(self) -> str:
        return f'{self.name}()'


def generate_random_graph(num_nodes, density, num_classes, feat_dim=64):
    edge_index = torch.randint(0, num_nodes, (2, int(num_nodes * density)), dtype=torch.long)
    x = torch.randn((num_nodes, feat_dim))  # Features (random in this case)
    y = torch.randint(0, num_classes, (num_nodes,))  # Random labels

    data = Data(x=x, edge_index=edge_index, y=y)
    data = T.NormalizeFeatures()(data)  # Normalize node features
    return data


def get_dataset(name, root='./data', seed=None, num_nodes=None, density=None, num_classes=None, feat_dim=None):
    transform = T.NormalizeFeatures()
    if name in ('cora', 'citeseer', 'pubmed'):
        return Planetoid(root, name, transform=transform)
    elif name == 'chameleon':
        return WikipediaNetwork(root, name, transform=transform)
    elif name in ('cornell', 'wisconsin', 'texas'):
        return WebKB(root, name, transform=transform)
    elif name == "random":
        assert seed is not None
        assert num_nodes is not None
        assert density is not None
        assert num_classes is not None
        assert feat_dim is not None
        return RandomGraphDataset(root, f"{name}_seed{seed}_{num_nodes}_{density}_{num_classes}_{feat_dim}", num_nodes=num_nodes, density=density, num_classes=num_classes, feat_dim=feat_dim)
    else:
        raise ValueError("Dataset not found or not supported.")


if __name__ == '__main__':
    # Set the path where the dataset will be stored
    root = 'data'
    # Ensure the directory exists before saving
    os.makedirs(root, exist_ok=True)

    num_nodes = 100
    density = 0.1  # Density of the graph
    num_classes = 5

    # Create the dataset
    dataset = RandomGraphDataset(root, num_nodes, density, num_classes)
