import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np


class AE(nn.Module):
    def __init__(self, dim, emb_dim=128):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, emb_dim)
        self.fc3 = nn.Linear(emb_dim, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z


def reduction_AE(gene_cell, device, use_amp=True):
    gene = torch.tensor(gene_cell, dtype=torch.float32).to(device)
    if gene_cell.shape[0] < 5000:
        ba = gene_cell.shape[0]
    else:
        ba = 5000
    gene_embed = train_AE(gene, ba, device, use_amp=use_amp)

    if gene_cell.shape[1] < 5000:
        ba = gene_cell.shape[1]
    else:
        ba = 5000
    cell = torch.tensor(np.transpose(gene_cell),
                        dtype=torch.float32).to(device)
    cell_embed = train_AE(cell, ba, device, use_amp=use_amp)
    return gene_embed, cell_embed


def train_AE(feature, ba, device, alpha=0.5, is_init=False, use_amp=True):
    model = AE(dim=feature.shape[1]).to(device)
    model.train()
    
    loader = Data.DataLoader(feature, ba)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_func = nn.MSELoss()
    use_amp = use_amp and device.type == 'cuda'
    EPOCH_AE = 2000
    for epoch in range(EPOCH_AE):
        embeddings = []
        # loss_ls=[]
        for _, batch_x in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                decoded, encoded = model(batch_x)
                loss = loss_func(batch_x, decoded)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            embeddings.append(encoded.float())
        #     loss_ls.append(loss.item())
        # scheduler.step(np.mean(loss_ls))
    print('Epoch :', epoch, '|', 'train_loss:%.12f' % loss.data)
    return torch.cat(embeddings)
