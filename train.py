import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch import nn
import tqdm
import os

from model import *

IM_SIZE = 32

batch_size = 64
epochs = 10

train = pneumonia_dataset("/home/wcsng/hwk/chest_xray/easy/train", 32)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

feat, lab = next(iter(train_loader))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

en = encoder(IM_SIZE*IM_SIZE,256,8)
de = decoder(8,256,IM_SIZE*IM_SIZE)
mo = model(en,de,device).to(device)


from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss(x,x_hat,mean,log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat,x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = Adam(mo.parameters(), lr=1e-3)

print("starting")

mo.train()
for epoch in range(epochs):
    ov_loss = 0
    for batch_idx, (x,_) in tqdm.tqdm(enumerate(train_loader)):
        x = x.view(x.shape[0], IM_SIZE * IM_SIZE)
        x = x.to(device)

        optimizer.zero_grad()
        x_hat, mean, log_var = mo(x)
        l=loss(x,x_hat,mean,log_var)
        ov_loss += l
        l.backward()

        optimizer.step()

    print(f"\tEpoch:{epoch}, avg loss {ov_loss/(batch_idx*batch_size)}\n")

torch.save(mo.state_dict(), "easy.pt")    
torch.save(mo.encoder.state_dict(), "easy_en.pt")
torch.save(mo.decoder.state_dict(), "easy_de.pt")
# import matplotlib.pyplot as plt
# for i in range(8):
#     fig = plt.figure()
#     for k in range(8):
#         gnd = fig.add_subplot(8,2,2*k + 1)
#         gnd.imshow(x[8*i + k,:].cpu().detach().numpy().reshape((IM_SIZE,IM_SIZE)))
#         gnd.axis('off')
#         pred = fig.add_subplot(8,2,2*k + 2)
#         pred.imshow(x_hat[8*i + k,:].cpu().detach().numpy().reshape((IM_SIZE,IM_SIZE)))
#         pred.axis('off')

#     plt.show()

