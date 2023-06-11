import numpy as np
from PIL import Image

size=32
xsp = np.linspace(-10,10,size)
ysp = np.linspace(-10,10,size)

imsp = np.stack(np.meshgrid(xsp,ysp)).transpose(2,1,0)[:,:,:,np.newaxis]

def gen(mu,sig):
    ret = np.exp(-1.0 * ((imsp-mu).transpose(0,1,3,2) @ sig @ (imsp-mu)))
    return (((ret*(1.0/np.max(ret))).squeeze())*255).astype(np.uint8)

import tqdm
N=3000

dset="test"
nor = f"/home/wcsng/hwk/chest_xray/easy/{dset}/NORMAL"
pne = f"/home/wcsng/hwk/chest_xray/easy/{dset}/PNEUMONIA"
#normal

N_MEAN = np.asarray([[4.0,0.0]]).T
P_MEAN = np.asarray([[-4.0,0.0]]).T

for i in tqdm.trange(N):
    mu = np.random.normal(N_MEAN, 1.0)
    sig = np.random.normal(0.0,0.1,size=(2,1))
    sig = 0.1*np.identity(2) + sig @ sig.T
    im = gen(mu, sig)
    Image.fromarray(im, "L").save(f"{nor}/{i}.jpg")
for i in tqdm.trange(N):
    mu = np.random.normal(P_MEAN, 1.0)
    sig = np.random.normal(0.0,0.1,size=(2,1))
    sig = 0.1*np.identity(2) + sig @ sig.T
    im = gen(mu, sig)
    Image.fromarray(im, "L").save(f"{pne}/{i}.jpg")
