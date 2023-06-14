import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch import nn
import tqdm
import os


class pneumonia_dataset(Dataset):
    def __init__(self, path, im_size=32):
        self.labels = []
        self.paths = []
        self.tf = transforms.Compose([
            transforms.Resize(size=(im_size,im_size)),
            transforms.Grayscale(),
            transforms.ConvertImageDtype(torch.float32)
        ])

        normal = os.path.join(path, "NORMAL")
        for p in os.listdir(normal):
            self.labels.append(0)
            self.paths.append(os.path.join(normal,p))

        pneumonia = os.path.join(path, "PNEUMONIA")
        for p in os.listdir(pneumonia):
            self.labels.append(1)
            self.paths.append(os.path.join(pneumonia,p))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tf(read_image(self.paths[idx])), self.labels[idx]
            

class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(encoder, self).__init__()
        self.flatten = torch.nn.Flatten
        self.lin_1 = nn.Linear(input_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.FC_class = nn.Linear(hidden_dim, 1)
        self.LeakyRelu = nn.LeakyReLU(0.2)
        self.training = True
        
    def forward(self, x):
        h_ = self.LeakyRelu(self.lin_1(x))
        h_ = self.LeakyRelu(self.lin_2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        prob_logit = self.FC_class(h_)
        return mean, log_var, prob_logit

class decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(decoder, self).__init__()
        self.lin_1 = nn.Linear(latent_dim, hidden_dim)
        self.lin_prob = nn.Linear(1,hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, output_dim)
        self.lin_prob = nn.Linear(1, hidden_dim)
        self.LeakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x, prob):
        h = self.LeakyRelu(self.lin_1(x) + self.lin_prob(prob))
        h = self.LeakyRelu(self.lin_2(h))
        x_hat = torch.sigmoid(self.lin_out(h))
        return x_hat

class model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, log_var, prob_logit = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5*log_var))
        x_hat = self.decoder(z, torch.sigmoid(prob_logit))

        return x_hat, mean, log_var, prob_logit

