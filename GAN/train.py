from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import os

from gan import GAN


n_epoch = 1000
save_result = 10

device = torch.device("cuda:1")
model = GAN(device = device)

tmp_folder = "tmp"
os.makedirs(tmp_folder, exist_ok= True)

train_data= MNIST("../../Data/", train = True, download=True, transform = T.ToTensor())
train_loader = DataLoader(train_data, batch_size= 1024)

optimizer_D = Adam(model.D.parameters(), lr = 1e-4)
optimizer_G = Adam(model.G.parameters(), lr = 1e-4)

pbar = tqdm(range(n_epoch))
loss_plot = {}
fixed_sample= torch.rand(10,10)
for ep in pbar:
    loss_dict = model.train(train_loader, optimizer_D, optimizer_G)
    print_str= ""
    for key in loss_dict.keys():
        if not key in loss_plot.keys():
            loss_plot[key] = []
        print_str += f"{key}={loss_dict[key]:.4f} "
        loss_plot[key].append(loss_dict[key])
    pbar.set_description_str(f"{ep} {print_str}")
    if ep % save_result == 0:
        gen = model.example(fixed_sample).view(-1, 1, 28, 28)
        save_image(gen, f"{tmp_folder}/{ep}_save.png" ,nrow = 5)
for key in loss_plot.keys():
    plt.plot(loss_plot[key],label=key)
    plt.legend()
plt.savefig("./loss.png")