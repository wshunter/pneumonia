import torch
from model import *
import matplotlib.pyplot as plt
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

IM_SIZE=32
def load(path):
    en = encoder(IM_SIZE*IM_SIZE,256,8)#.load_state_dict(torch.load(f"{path}_en.pt"))
    de = decoder(8,256,IM_SIZE*IM_SIZE)#.load_state_dict(torch.load(f"{path}_de.pt"))
    mo = model(en,de,device).to(device)
    mo.load_state_dict(torch.load(f"{path}.pt"))
    return mo

mod = load("/home/wcsng/hwk/chest_xray/easy")
mod.eval()

with torch.no_grad():
    point = torch.randn(8).to(device)
    im = mod.decoder.forward(point).cpu().numpy().reshape((32,32))

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(im)
plt.show()
