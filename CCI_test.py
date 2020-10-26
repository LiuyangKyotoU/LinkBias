import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    average_precision_score, roc_auc_score, f1_score, auc, precision_score, recall_score

from CCI_links import get_CCI_links
from models import Net_cls

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int)
parser.add_argument('--alpha', type=float)
args = parser.parse_args()

times = args.times
alpha = args.alpha

results_path = 'results/CCI/gap_' + str(alpha) + '_' + str(times) + '.txt'

pos_links, neg_links = get_CCI_links()
mask_pos = torch.load('mask/gap_CCI_e_pos_' + str(alpha) + '_' + str(times) + '.pt')
mask_neg = torch.load('mask/gap_CCI_e_neg_' + str(alpha) + '_' + str(times) + '.pt')
idx_pos = (mask_pos == 3).nonzero().squeeze().tolist()
idx_neg = (mask_neg == 3).nonzero().squeeze().tolist()
test_data = [pos_links[i] for i in idx_pos] + [neg_links[i] for i in idx_neg]

test_loader = DataLoader(test_data, batch_size=64, shuffle=False, follow_batch=['x_s', 'x_t'])

for relative in [0, 0.25, 0.5, 0.75, 1]:
    predictor_path = 'results/CCI/predictor/gap_CCI_e_alpha' + str(relative) + '_' + str(alpha) + '_' + str(times)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = Net_cls().to(device)
    predictor = torch.load(predictor_path + '.pt')
    predictor.eval()

    pred, truth = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred += (predictor(data).max(1)[1]).cpu().tolist()
            truth += data.y.cpu().tolist()
    pred = np.array(pred)
    truth = np.array(truth)

    with open(results_path, 'a') as f:
        if relative == 0.0:
            f.write('acc' + '\t' +
                    'b_acc' + '\t' +
                    'ap' + '\t' +
                    'precision' + '\t' +
                    'recall' + '\t' +
                    'roc_auc' + '\t' +
                    'auc' + '\t' +
                    'f1' + '\t' +
                    'f1_micro' + '\t' +
                    'f1_macro' + '\n'
                    )

        f.write(str(accuracy_score(truth, pred)) + '\t' +
                str(balanced_accuracy_score(truth, pred)) + '\t' +
                str(average_precision_score(truth, pred)) + '\t' +
                str(precision_score(truth, pred)) + '\t' +
                str(recall_score(truth, pred)) + '\t' +
                str(roc_auc_score(truth, pred)) + '\t' +
                str(auc(truth, pred)) + '\t' +
                str(f1_score(truth, pred)) + '\t' +
                str(f1_score(truth, pred, average='micro')) + '\t' +
                str(f1_score(truth, pred, average='macro')) + '\n'
                )

