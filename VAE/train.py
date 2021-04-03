import torch
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import os

from vae import VAE


n_epoch = 1000
save_result = 10

device = torch.device("cuda:0")
model = VAE(device = device)

tmp_folder = "tmp"
os.makedirs(tmp_folder, exist_ok= True)

train_data= MNIST("../../Data", train = True, download=True, transform = T.ToTensor())
train_loader = DataLoader(train_data, batch_size= 1024)

optimizer = Adam(model.parameters(), lr = 1e-4)

pbar = tqdm(range(n_epoch))
loss_plot = {}
fixed_sample = {}
for idx, (img, labels) in enumerate(train_loader):
    for idx, label in enumerate(labels):
        if not label.item() in fixed_sample.keys():
            fixed_sample[label.item()] = img[idx]
fixed_sample = torch.stack(list(fixed_sample.values()), dim =0)
save_image(fixed_sample, f"{tmp_folder}/origin.png" ,nrow = 5)
for ep in pbar:
    loss_dict = model.train_epoch(train_loader, optimizer)
    print_str= ""
    for key in loss_dict.keys():
        if not key in loss_plot.keys():
            loss_plot[key] = []
        print_str += f"{key}={loss_dict[key]:.4f} "
        loss_plot[key].append(loss_dict[key])
    pbar.set_description_str(f"{ep} {print_str}")
    if ep % save_result == 0:
        mu, sigma = model.encode(fixed_sample)
        gen = model.decode(mu)
        save_image(gen, f"{tmp_folder}/{ep}_save.png" ,nrow = 5)
for key in loss_plot.keys():
    plt.plot(loss_plot[key],label=key)
    plt.legend()
plt.savefig("./loss.png")