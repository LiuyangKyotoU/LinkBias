import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse

from CCI_links import get_CCI_links
from models import Net_cls

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int)
parser.add_argument('--alpha', type=float)
args = parser.parse_args()

times = args.times
alpha = args.alpha

pos_links, neg_links = get_CCI_links()
mask_pos = torch.load('mask/gap_CCI_e_pos_' + str(alpha) + '_' + str(times) + '.pt')
mask_neg = torch.load('mask/gap_CCI_e_neg_' + str(alpha) + '_' + str(times) + '.pt')
idx_pos_train = (mask_pos == 1).nonzero().squeeze().tolist()
idx_pos_val = (mask_pos == 2).nonzero().squeeze().tolist()
idx_neg_train = (mask_neg == 1).nonzero().squeeze().tolist()
idx_neg_val = (mask_neg == 2).nonzero().squeeze().tolist()
train_data = [pos_links[i] for i in idx_pos_train] + [neg_links[i] for i in idx_neg_train]
val_data = [pos_links[i] for i in idx_pos_val] + [neg_links[i] for i in idx_neg_val]
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, follow_batch=['x_s', 'x_t'])
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, follow_batch=['x_s', 'x_t'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier_path = 'results/CCI/classifier/gap_CCI_e_' + str(alpha) + '_' + str(times) + '.pt'
classifier = Net_cls().to(device)
classifier.load_state_dict(torch.load(classifier_path))
classifier.eval()

for relative in [0, 0.25, 0.5, 0.75, 1]:
    predictor_path = 'results/CCI/predictor/gap_CCI_e_alpha' + str(relative) + '_' + str(alpha) + '_' + str(times)
    print(predictor_path)

    predictor = Net_cls().to(device)
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
                weights = weights.unsqueeze(1).data
                loss = F.nll_loss(weights * predictor(data), data.y)
            else:
                loss = F.nll_loss(predictor(data), data.y)
            loss.backward()
            optimizer.step()
        predictor.eval()
        val_correct = 0
        for data in val_loader:
            data = data.to(device)
            pred = predictor(data).max(1)[1]
            val_correct += pred.eq(data.y).sum().item()
        val_correct = val_correct / len(val_loader.dataset)
        scheduler.step(1 - val_correct)
        if best_val_correct is None or 1 - val_correct <= best_val_correct:
            torch.save(predictor, predictor_path + '.pt')
        with open(predictor_path + '.txt', 'a') as f:
            f.write(str(epoch) + '\t' + str(val_correct) + '\n')
        print(epoch, val_correct)
