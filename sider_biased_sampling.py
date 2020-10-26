import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sider_graphs import sider_graphs
from sider_links import get_sider_links

parser = argparse.ArgumentParser()
# parser.add_argument('--times', type=int)
parser.add_argument('--alpha', type=float)
args = parser.parse_args()
# times = args.times
alpha = args.alpha

graphs = sider_graphs('data/DDI')
links = get_sider_links()


def check_sigmoid(x1, x2, xr, a, c, mode):
    x = np.arange(x1, x2, xr)
    y = 1 / (1 + np.exp(-a * (x - c)))

    fig, ax = plt.subplots()
    if mode == '-':
        ax.plot(x, 1 - y)
    else:
        ax.plot(x, y)
    plt.show()


def calculate_sigmoid(x, a, c):
    y = 1 / (1 + np.exp(-a * (x - c)))
    return y


l = torch.Tensor([g.x.shape[0] for g in graphs])


def sigmoid_fig():
    x = np.arange(l.min().item(), l.mean().item() * 2 - l.min().item(), 1)

    fig, ax1 = plt.subplots()
    ax1.hist(l, weights=np.ones_like(l) / len(l), bins=100, alpha=0.5)
    ax1.set_xlim([0, 100])
    ax2 = ax1.twinx()
    ax2.plot(x, 1 - 1 / (1 + np.exp(-0.5 * (x - 25))))


for times in range(10):

    l_sig = 1 - calculate_sigmoid(l, alpha, 22)
    l_sig = l_sig - torch.randint(0, 100, l_sig.size()).to(torch.float) / 100
    idx = (l_sig >= 0).nonzero().squeeze()

    graphs_obs = graphs[idx]
    print('All:{}  Observed nodes:{}'.format(graphs, graphs_obs))

    mask_v = torch.zeros(len(graphs)).to(torch.long)
    mask_v[idx] = 1  # unobs: 0, obs: 1

    mask_e = torch.zeros(len(links)).to(torch.long)
    mask_e[torch.randperm(len(links))[:len(links) // 10]] = 3 # test: 3
    print('test size: {}'.format((mask_e == 3).nonzero().shape))
    for i, link in enumerate(links):
        if mask_v[link.idxs[0][0]] == 1 and mask_v[link.idxs[0][1]] == 1:
            if mask_e[i] == 0:
                mask_e[i] = 1

    tmp = (mask_e == 1).nonzero()
    tmp = tmp[torch.randperm(tmp.shape[0])][:tmp.shape[0] // 5]
    mask_e[tmp] = 2
    print('train size: {}'.format((mask_e == 1).nonzero().shape))
    print('val size: {}'.format((mask_e == 2).nonzero().shape))
    print('unobs size: {}'.format((mask_e == 0).nonzero().shape))

    torch.save(mask_v, 'mask/sider_v_' + str(alpha) + '_' + str(times) + '.pt')
    torch.save(mask_e, 'mask/sider_e_' + str(alpha) + '_' + str(times) + '.pt')
