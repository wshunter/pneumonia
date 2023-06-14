import torch
from model_classifier import *
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#device = "cpu"
print(f"Using {device} device")

IM_SIZE=224
HIDDEN_DIM=256
LATENT_DIM=32
def load(path):
    en = encoder(IM_SIZE*IM_SIZE,HIDDEN_DIM,LATENT_DIM)#.load_state_dict(torch.load(f"{path}_en.pt"))
    de = decoder(LATENT_DIM,HIDDEN_DIM,IM_SIZE*IM_SIZE)#.load_state_dict(torch.load(f"{path}_de.pt"))
    mo = model(en,de,device).to(device)
    mo.load_state_dict(torch.load(f"{path}.pt"))
    return mo

mod = load("/home/wcsng/hwk/chest_xray/real_deep")
mod.eval()

test_data = pneumonia_dataset("/home/wcsng/hwk/chest_xray/real/test", 224)
loader = DataLoader(test_data, batch_size=64, shuffle=True)


raw = []
pred = []
gnd = []
x_pred = []
x_gnd = []
with torch.no_grad():
    for batch_idx, (x,label) in tqdm.tqdm(enumerate(test_data)):
        x_gnd.append(x)
        gnd.append(label)
        x=x.view(x.shape[0],IM_SIZE*IM_SIZE)
        xhat,mean,var,logit = mod.forward(x)
        x_pred.append(xhat.numpy())
        p = torch.sigmoid(logit[0,0])
        raw.append(p.item())
        if p.item() > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    
import numpy as np
pred = np.asarray(pred)
gnd = np.asarray(gnd)

acc = 1 - float(np.count_nonzero(pred != gnd))/gnd.shape[0]
print(f"Classification Accuracy: {acc:.3f}")

fig = plt.figure(figsize=(6,9))

with torch.no_grad():
    for i in range(3):
        ax=fig.add_subplot(3,2,2*i+1)
        point = torch.randn(32).to(device)
        cl_prob = torch.tensor([-10.0]).to(device)
        im = mod.decoder.forward(point, cl_prob).cpu().numpy().reshape((224,224))
        ax.imshow(im, cmap='Greys')
        ax.axis('off')
        if i == 0:
            ax.set_title("Normal")
    for i in range(3):
        ax=fig.add_subplot(3,2,2*i+2)
        point = torch.randn(32).to(device)
        cl_prob = torch.tensor([11.0]).to(device)
        im = mod.decoder.forward(point, cl_prob).cpu().numpy().reshape((224,224))
        ax.imshow(im, cmap='Greys')
        ax.axis('off')
        if i == 0:
            ax.set_title("Pneumonia")

loader = DataLoader(test_data, batch_size=5, shuffle=True)
fig2 = plt.figure(figsize=(15,6))
with torch.no_grad():
    x,lab = next(iter(loader))
    x=x.view(x.shape[0],IM_SIZE*IM_SIZE)
    xh,_,_,_ = mod.forward(x)

    for i in range(5):
        ax = fig2.add_subplot(2,5,i+1)
        ax.imshow(x[i,:].view(IM_SIZE,IM_SIZE).cpu().numpy(),cmap='Greys')
        ax.axis('off')
        ax = fig2.add_subplot(2,5,i+6)
        ax.imshow(xh[i,:].view(IM_SIZE,IM_SIZE).cpu().numpy(),cmap='Greys')
        ax.axis('off')
fig2.set_tight_layout(True)
            
fig.set_tight_layout(True)

x_pred = np.stack(x_pred).squeeze()
x_pred_t = torch.Tensor(x_pred)
x_pred_t = x_pred_t.view(x_pred_t.shape[0],IM_SIZE,IM_SIZE)

x_gnd = np.stack(x_gnd).squeeze()
x_gnd_t = torch.Tensor(x_gnd)
x_gnd_t = x_gnd_t.view(x_gnd_t.shape[0],IM_SIZE,IM_SIZE)


import torchvision
resize = torchvision.transforms.Resize((299,299))
x_pred_t = resize(x_pred_t)
x_gnd_t = resize(x_gnd_t)

x_gnd_t = x_gnd_t.view(624,1,299,299).expand(624,3,299,299)
x_pred_t = x_pred_t.view(624,1,299,299).expand(624,3,299,299)

import ignite
from ignite.metrics import FID, InceptionScore
from ignite.engine import *
fid_metric = FID(device=device)
is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])

def eval_step(engine, batch):
    return batch
default_evaluator = Engine(eval_step)
fid_metric.attach(default_evaluator, "fid")
is_metric.attach(default_evaluator, "is")

state = default_evaluator.run([[x_pred_t, x_gnd_t]])
print(state)

#plt.show()


