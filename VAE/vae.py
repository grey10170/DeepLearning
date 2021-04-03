import pickle
from torch import optim
import torch.nn as nn
import torch

class View(nn.Module):
    def __init__(self, view):
        super().__init__()
        self.view =view
    def forward(self, x):
        return x.view(self.view)

class VAE(nn.Module):
    def __init__(self, input_size = (28,28), latent_dim = 10 ,device = None):
        super().__init__()
        self.encoder = nn.Sequential(
            View((-1, input_size[0]*input_size[1])),
            nn.Linear(input_size[0]*input_size[1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2*latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512 , input_size[0]*input_size[1]),
            nn.Sigmoid(),
            View((-1, 1, input_size[0], input_size[1]))
        )
        self.latent_dim = latent_dim
        self.device = device
        if self.device is not None:
            self.encoder.to(self.device)
            self.decoder.to(self.device)
    def train_epoch(self, train_loader, optimizer):
        #training of 1 epoch
        #train_loader will give (b,1,28,28)
        loss_dict= {}
        for key in ["NLL", "KL", "total"]:
            loss_dict[key] = 0
        for idx, (img, _ ) in enumerate(train_loader):
            batch_size = img.shape[0]
            img = img.to(self.device)
            optimizer.zero_grad()
            latent = self.encoder(img.view(batch_size, -1))
            mu, sigma = latent[:,:self.latent_dim], latent[:,self.latent_dim:]
            input_ = mu+sigma*torch.randn((batch_size, self.latent_dim), device= self.device)
            gen_img = self.decoder(input_)
            nll = nn.BCELoss(reduction= 'sum')(gen_img, img)/batch_size #Berniouli 가정으로 log값을 모두 더한다.
            kl = torch.mean(torch.sum(mu**2 + sigma**2-1-torch.log(sigma **2), dim = 1)/2)
            loss = nll+ kl
            # print(loss.item())
            loss.backward()
            optimizer.step()
            loss_dict["NLL"] += nll.item()
            loss_dict["KL"] += kl.item()
            loss_dict["total"] +=loss.item()
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key]/ len(train_loader)
        return loss_dict
    @torch.no_grad()
    def encode(self, img):
        img = img.to(self.device)
        latent_code = self.encoder(img)
        return latent_code[:,:self.latent_dim], latent_code[:,self.latent_dim:]
    @torch.no_grad()
    def decode(self, latent_code):
        latent_code = latent_code.to(self.device)
        img = self.decoder(latent_code)
        return img