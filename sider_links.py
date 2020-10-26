import torch
from torch_geometric.data import Data
from tqdm import tqdm

from sider_graphs import sider_graphs


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


def get_sider_links():
    graphs = sider_graphs('data/DDI')
    with open('data/DDI/raw/sider_links.txt', 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
    # for line in tqdm(lines):
        line = line.split()
        g_s = graphs[int(line[0])]
        g_t = graphs[int(line[1])]
        idxs = torch.LongTensor([[int(line[0]), int(line[1])]])
        y = torch.FloatTensor([float(line[2])])
        data.append(PairData(
            edge_index_s=g_s.edge_index,
            edge_attr_s=g_s.edge_attr,
            x_s=g_s.x,
            edge_index_t=g_t.edge_index,
            edge_attr_t=g_t.edge_attr,
            x_t=g_t.x,
            idxs=idxs,
            y=y,
        ))
    return data