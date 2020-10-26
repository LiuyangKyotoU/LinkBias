import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse

from CCI_links import get_CCI_links
from models import Net_cls

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--mode', type=str)
args = parser.parse_args()

times = args.times
alpha = args.alpha
mode = args.mode

print(mode, alpha, times)

classifier_path = 'results/CCI/classifier/CCI_e_' + mode + '_' + str(alpha) + '_' + str(times) + '.pt'
predictor_path = 'results/CCI/predictor/CCI_e_ipw' + '_' + mode + '_' + str(alpha) + '_' + str(times)

pos_links, neg_links = get_CCI_links()
mask_pos = torch.load('mask/CCI_e_pos_' + str(alpha) + '_' + str(times) + '.pt')
mask_neg = torch.load('mask/CCI_e_neg_' + str(alpha) + '_' + str(times) + '.pt')
idx_pos_train = (mask_pos == 1).nonzero().squeeze().tolist()
idx_pos_val = (mask_pos == 2).nonzero().squeeze().tolist()
idx_neg_train = (mask_neg == 1).nonzero().squeeze().tolist()[:len(idx_pos_train)]
idx_neg_val = (mask_neg == 2).nonzero().squeeze().tolist()[:len(idx_pos_val)]
train_data = [pos_links[i] for i in idx_pos_train] + [neg_links[i] for i in idx_neg_train]
val_data = [pos_links[i] for i in idx_pos_val] + [neg_links[i] for i in idx_neg_val]
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, follow_batch=['x_s', 'x_t'])
val_loader = DataLoader(val_data, batch_size=128, shuffle=False, follow_batch=['x_s', 'x_t'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = Net_cls().to(device)
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

classifier = Net_cls().to(device)
classifier.load_state_dict(torch.load(classifier_path))
classifier.eval()

best_val_correct, best_val_loss = None, None
for epoch in range(1, 301):
    predictor.train()
    for data in train_loader:
        data = data.to(device)

        weights = 1 / torch.exp(classifier(data))[:, 1]
        weights = weights.unsqueeze(1).data

        optimizer.zero_grad()
        loss = F.nll_loss(weights * predictor(data), data.y)
        loss.backward()
        optimizer.step()
    predictor.eval()
    val_correct, val_loss = 0, 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)

            weights = 1 / torch.exp(classifier(data))[:, 1]
            weights = weights.unsqueeze(1).data

            val_loss += F.nll_loss(weights * predictor(data), data.y).item() * data.y.shape[0]
            val_correct += predictor(data).max(1)[1].eq(data.y).sum().item()
    val_correct = 1 - val_correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    if best_val_correct is None or val_correct <= best_val_correct:
        torch.save(predictor, predictor_path + '_bestacc.pt')
    if best_val_loss is None or val_loss <= best_val_loss:
        torch.save(predictor, predictor_path + '_bestunbiasloss.pt')
    with open(predictor_path + '.txt', 'a') as f:
        f.write(str(epoch) + '\t' + str(val_correct) + '\t' + str(val_loss) + '\n')
    print(epoch, val_correct, val_loss)
