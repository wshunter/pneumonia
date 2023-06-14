import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch import nn
import tqdm
import os
import numpy as np

from model_classifier import *

IM_SIZE = 224
HIDDEN_SIZE=512
LATENT_SIZE=32
batch_size = 64
epochs = 1000

train = pneumonia_dataset("/home/wcsng/hwk/chest_xray/real/train", IM_SIZE)
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

en = encoder(IM_SIZE*IM_SIZE,256,LATENT_SIZE)
de = decoder(LATENT_SIZE,256,IM_SIZE*IM_SIZE)
mo = model(en,de,device).to(device)


from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss(x,label,x_hat,mean,log_var,prob_logit):
    
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat,x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    print(prob_logit)
    classifier_error = 10.0*torch.sum((label-torch.sigmoid(prob_logit)).pow(2))
    return reproduction_loss + KLD  + classifier_error

optimizer = Adam(mo.parameters(), lr=1e-3)

print("starting")

mo.train()

losses = []
for epoch in range(epochs):
    ov_loss = 0
    for batch_idx, (x,label) in tqdm.tqdm(enumerate(train_loader)):
        x = x.view(x.shape[0], IM_SIZE * IM_SIZE)
        x = x.to(device)
        label=label[:,None].to(device)

        optimizer.zero_grad()
        x_hat, mean, log_var, prob_logit = mo(x)
        l=loss(x,label,x_hat,mean,log_var,prob_logit)
        ov_loss += l
        l.backward()

        optimizer.step()
    
    torch.save(mo.state_dict(), "real_deep.pt")
    losses.append((ov_loss/(batch_idx*batch_size)).cpu().detach())
    np.save("/home/wcsng/hwk/chest_xray/losses_deep.npy", np.asarray(losses))
    
    print(f"\tEpoch:{epoch}, avg loss {ov_loss/(batch_idx*batch_size)}\n")

torch.save(mo.state_dict(), "real_deep.pt")    
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

