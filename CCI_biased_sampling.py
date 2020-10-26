import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from CCI_graphs import CCI_graphs
from CCI_links import get_CCI_links

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int)
parser.add_argument('--alpha', type=float)
args = parser.parse_args()
times = args.times
alpha = args.alpha

graphs = CCI_graphs('data/CCI')
pos_links, neg_links = get_CCI_links()


def check_sigmoid(x1, x2, xr, a, c, mode):
    x = np.arange(x1, x2, xr)
    y = 1 / (1 + np.exp(-a * (x - c)))

    fig, ax = plt.subplots()
    if mode == '-':
        ax.plot(x, 1 - y)
    else:
        ax.plot(x, y)
    plt.show()


# T = [L[i] for i in Idx]

def calculate_sigmoid(x, a, c):
    y = 1 / (1 + np.exp(-a * (x - c)))
    return y


# indicator: weight
l = torch.Tensor([g.weight.item() for g in graphs])
# plt.hist(l, weights=np.ones_like(l) / len(l), bins=1000, alpha=0.5)
# plt.xlim([0, 1000])
# check_sigmoid(l.min().item(), l.mean().item() * 2 - l.min().item(), 1, 0.04,  l.mean().item(), '-')
l_sig = 1 - calculate_sigmoid(l, alpha, l.mean().item())
l_sig = l_sig - torch.randint(0, 100, l_sig.size()).to(torch.float) / 100
idx = (l_sig >= 0).nonzero().squeeze()

graphs_obs = graphs[idx]
print('All:{}  Observed nodes:{}'.format(graphs, graphs_obs))
mask_v = torch.zeros(len(graphs)).to(torch.long)
mask_v[idx] = 1  # unobs: 0, obs: 1

mask_e_pos = torch.zeros(len(pos_links)).to(torch.long)
# mask_e_pos[torch.randperm(mask_e_pos.shape[0])[:20000]]=3 # test: 3
for i, link in enumerate(pos_links):
    if mask_v[link.idxs[0][0]] == 1 and mask_v[link.idxs[0][1]] == 1:
        if mask_e_pos[i] == 0:
            mask_e_pos[i] = 1
print('pos size: {}'.format((mask_e_pos == 1).nonzero().shape))
tmp = (mask_e_pos == 1).nonzero()
tmp = tmp[torch.randperm(tmp.shape[0])][:round(tmp.shape[0] / 5)]
mask_e_pos[tmp] = 2

mask_e_neg = torch.zeros(len(neg_links)).to(torch.long)
for i, link in enumerate(neg_links):
    if mask_v[link.idxs[0][0]] == 1 and mask_v[link.idxs[0][1]] == 1:
        if mask_e_neg[i] == 0:
            mask_e_neg[i] = 1
print('neg size: {}'.format((mask_e_neg == 1).nonzero().shape))
tmp = (mask_e_neg == 1).nonzero()
tmp = tmp[torch.randperm(tmp.shape[0])][:round(tmp.shape[0] / 5)]
mask_e_neg[tmp] = 2

torch.save(mask_v, 'mask/CCI_v_' + str(alpha) + '_' + str(times) + '.pt')
torch.save(mask_e_pos, 'mask/CCI_e_pos_' + str(alpha) + '_' + str(times) + '.pt')
torch.save(mask_e_neg, 'mask/CCI_e_neg_' + str(alpha) + '_' + str(times) + '.pt')
