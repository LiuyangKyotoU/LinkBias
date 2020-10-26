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

classifier_path = 'results/CCI/classifier/gap_CCI_e_' + str(alpha) + '_' + str(times) + '.pt'
mask_pos = torch.load('mask/gap_CCI_e_pos_' + str(alpha) + '_' + str(times) + '.pt')
mask_neg = torch.load('mask/gap_CCI_e_neg_' + str(alpha) + '_' + str(times) + '.pt')

idx_pos = (mask_pos == 1).nonzero().squeeze().tolist()
idx_neg = (mask_neg == 1).nonzero().squeeze().tolist()
classify_pos = [pos_links[i] for i in idx_pos] + [neg_links[i] for i in idx_neg]
print(len(classify_pos))
for link in classify_pos:
    link.y = torch.LongTensor([1])

idx_pos = (mask_pos == 3).nonzero().squeeze().tolist()
idx_neg = (mask_neg == 3).nonzero().squeeze().tolist()
classify_neg = [pos_links[i] for i in idx_pos] + [neg_links[i] for i in idx_neg]
print(len(classify_neg))
for link in classify_neg:
    link.y = torch.LongTensor([0])

dataset = classify_pos + classify_neg
tmp = torch.randperm(len(dataset))
val_dataset = [dataset[i] for i in tmp[:len(dataset) // 5].tolist()]
train_dataset = [dataset[i] for i in tmp[len(dataset) // 5:].tolist()]
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, follow_batch=['x_s', 'x_t'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, follow_batch=['x_s', 'x_t'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = Net_cls().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

best_val_error = None
for epoch in range(1, 201):
    classifier.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(classifier(data), data.y)
        loss.backward()
        optimizer.step()
    classifier.eval()
    val_correct = 0
    for data in val_loader:
        data = data.to(device)
        pred = classifier(data).max(1)[1]
        val_correct += pred.eq(data.y).sum().item()
    val_correct = val_correct / len(val_loader.dataset)
    scheduler.step(1 - val_correct)
    if best_val_error is None or 1 - val_correct <= best_val_error:
        best_val_error = 1 - val_correct
        torch.save(classifier.state_dict(), classifier_path)
    print(epoch, val_correct)
