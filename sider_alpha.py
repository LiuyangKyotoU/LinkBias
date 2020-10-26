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

# normalize y
ys = []
for i in links:
    ys.append(i.y.item())
ys = torch.FloatTensor(ys)
mean = ys.mean()
std = ys.std()
for i in links:
    i.y = (i.y - mean) / std

mask = torch.load('mask/gap_sider_e_' + str(alpha) + '_' + str(times) + '.pt')
idx_train = (mask == 1).nonzero().squeeze().tolist()
idx_val = (mask == 2).nonzero().squeeze().tolist()

train_data = [links[i] for i in idx_train]
val_data = [links[i] for i in idx_val]
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, follow_batch=['x_s', 'x_t'])
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, follow_batch=['x_s', 'x_t'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier_path = 'results/DDI/classifier/gap_sider_e' + str(alpha) + '_' + str(times) + '.pt'
classifier = Net_cls().to(device)
classifier.load_state_dict(torch.load(classifier_path))
classifier.eval()

for relative in [0, 0.25, 0.5, 0.75, 1]:
    predictor_path = 'results/DDI/predictor/gap_sider_e_alpha' + str(relative) + '_' + str(alpha) + '_' + str(times)
    print(predictor_path)

    predictor = Net_reg().to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

    best_val_correct = None
    for epoch in range(1, 301):
        predictor.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            if relative != 0:
                out = torch.exp(classifier(data))
                weights = out[:, 0] / ((1 - relative) * out[:, 0] + relative * out[:, 1])
                weights = torch.sqrt(weights).data
                loss = F.mse_loss(weights * predictor(data), weights * data.y)
            else:
                loss = F.mse_loss(predictor(data), data.y)

            loss.backward()
            optimizer.step()

        predictor.eval()
        val_error = 0
        for data in val_loader:
            data = data.to(device)
            val_error += (predictor(data) * std - data.y * std).abs().sum().item()
        val_error = val_error / len(val_loader.dataset)
        scheduler.step(val_error)
        if best_val_correct is None or val_error <= best_val_correct:
            torch.save(predictor, predictor_path + '.pt')
        with open(predictor_path + '.txt', 'a') as f:
            f.write(str(epoch) + '\t' + str(val_error) + '\n')
        print(epoch, val_error)
