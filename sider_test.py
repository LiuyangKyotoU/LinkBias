import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse

from sider_links import get_sider_links
from models import Net_cls, Net_reg

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int)
parser.add_argument('--alpha', type=float)
# parser.add_argument('--mode', type=str)
# parser.add_argument('--relative', type=float)
args = parser.parse_args()

times = args.times
alpha = args.alpha
# mode = args.mode
# relative = args.relative

links = get_sider_links()

times = args.times
alpha = args.alpha
# mode = args.mode
# relative = args.relative

links = get_sider_links()

ys = []
for i in links:
    ys.append(i.y.item())
ys = torch.FloatTensor(ys)
mean = ys.mean()
std = ys.std()
for i in links:
    i.y = (i.y - mean) / std

mask = torch.load('mask/gap_sider_e_' + str(alpha) + '_' + str(times) + '.pt')
idx_test = (mask == 3).nonzero().squeeze().tolist()

test_data = [links[i] for i in idx_test]
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, follow_batch=['x_s', 'x_t'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for relative in [0, 0.25, 0.5, 0.75, 1]:
    predictor_path = 'results/DDI/predictor/gap_sider_e_alpha' + str(relative) + '_' + str(alpha) + '_' + str(times)
    predictor = Net_reg().to(device)
    predictor = torch.load(predictor_path + '.pt')
    predictor.eval()
    test_error = 0
    for data in test_loader:
        data = data.to(device)
        test_error += (predictor(data) * std - data.y * std).abs().sum().item()
    test_error = test_error / len(test_loader.dataset)
    with open('results/DDI/gap_sider_' + str(alpha) + '_' + str(times) + '.txt', 'a') as f:
        f.write(str(test_error) + '\n')
