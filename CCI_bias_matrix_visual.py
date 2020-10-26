import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

from CCI_graphs import CCI_graphs
from CCI_links import get_CCI_links

graphs = CCI_graphs('data/CCI')
_, indices = graphs.data.weight.sort()

# universe
mat = np.zeros((len(graphs), len(graphs)))
pos_links, neg_links = get_CCI_links()
links = pos_links + neg_links
for link in tqdm(links):
    idx1, idx2 = link.idxs[0][0].item(), link.idxs[0][1].item()
    mat_idx1, mat_idx2 = (indices == idx1).nonzero().item(), (indices == idx2).nonzero().item()
    mat[mat_idx1, mat_idx2] = 1
    mat[mat_idx2, mat_idx1] = 1
plt.spy(mat, markersize=0.00000005)
plt.xticks([0, 80000], fontsize=24)
plt.yticks([])

# unbiased
mat = np.zeros((len(graphs), len(graphs)))
pos_links, neg_links = get_CCI_links()
links = pos_links + neg_links
links = [links[i] for i in torch.randperm(len(links))[:70000].tolist()]
for link in tqdm(links):
    idx1, idx2 = link.idxs[0][0].item(), link.idxs[0][1].item()
    mat_idx1, mat_idx2 = (indices == idx1).nonzero().item(), (indices == idx2).nonzero().item()
    mat[mat_idx1, mat_idx2] = 1
    mat[mat_idx2, mat_idx1] = 1
plt.spy(mat, markersize=0.000001)
plt.xticks([0, 80000], fontsize=24)
plt.yticks([])

# biased
alpha = 0.16
mat = np.zeros((len(graphs), len(graphs)))
pos_links, neg_links = get_CCI_links()
mask_pos = torch.load('mask/CCI_e_pos_' + str(alpha) + '_0.pt')
mask_neg = torch.load('mask/CCI_e_neg_' + str(alpha) + '_0.pt')
idx_pos = (mask_pos == 1).nonzero().squeeze().tolist() + (mask_pos == 2).nonzero().squeeze().tolist()
idx_neg = (mask_neg == 1).nonzero().squeeze().tolist() + (mask_neg == 2).nonzero().squeeze().tolist()
links = [pos_links[i] for i in idx_pos] + [neg_links[i] for i in idx_neg]
for link in tqdm(links):
    idx1, idx2 = link.idxs[0][0].item(), link.idxs[0][1].item()
    mat_idx1, mat_idx2 = (indices == idx1).nonzero().item(), (indices == idx2).nonzero().item()
    mat[mat_idx1, mat_idx2] = 1
    mat[mat_idx2, mat_idx1] = 1
plt.spy(mat, markersize=0.000001)
plt.xticks([0, 80000], fontsize=24)
plt.yticks([])
