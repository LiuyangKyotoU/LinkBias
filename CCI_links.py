import torch
from torch_geometric.data import Data
from tqdm import tqdm

from CCI_graphs import CCI_graphs


class PairData(Data):
    def __init__(self, edge_index_s, edge_attr_s, x_s, edge_index_t, edge_attr_t, x_t,
                 idxs, y):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        # self.weight_s = weight_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.x_t = x_t
        # self.weight_t = weight_t
        self.idxs = idxs
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


def get_CCI_links():
    graphs = CCI_graphs('data/CCI')
    pos_pair_list, neg_pair_list = [], []
    with open('data/CCI/raw/pos_links_directed.txt', 'r') as f:
        lines = f.readlines()
    # for line in tqdm(lines):
    for line in lines:
        line = line.split()
        idx_s = int(line[0])
        idx_t = int(line[1])
        g_s = graphs[idx_s]
        g_t = graphs[idx_t]
        idxs = torch.LongTensor([[idx_s, idx_t]])
        y = torch.LongTensor([1])

        pos_pair_list.append(PairData(
            edge_index_s=g_s.edge_index,
            edge_attr_s=g_s.edge_attr,
            x_s=g_s.x,
            # weight_s=g_s.weight,
            edge_index_t=g_t.edge_index,
            edge_attr_t=g_t.edge_attr,
            x_t=g_t.x,
            # weight_t=g_t.weight,
            idxs=idxs,
            y=y,
        ))
    with open('data/CCI/raw/neg_links_directed.txt', 'r') as f:
        lines = f.readlines()
    # for line in tqdm(lines):
    for line in lines:
        line = line.split()
        idx_s = int(line[0])
        idx_t = int(line[1])
        g_s = graphs[idx_s]
        g_t = graphs[idx_t]
        idxs = torch.LongTensor([[idx_s, idx_t]])
        y = torch.LongTensor([0])

        neg_pair_list.append(PairData(
            edge_index_s=g_s.edge_index,
            edge_attr_s=g_s.edge_attr,
            x_s=g_s.x,
            # weight_s=g_s.weight,
            edge_index_t=g_t.edge_index,
            edge_attr_t=g_t.edge_attr,
            x_t=g_t.x,
            # weight_t=g_t.weight,
            idxs=idxs,
            y=y,
        ))
    return pos_pair_list, neg_pair_list
