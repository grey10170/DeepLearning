from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
import numpy as np
import random

from tqdm import tqdm
import os

from cgan import CGAN


n_epoch = 50
save_result = 1

random_seed = 10170
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda:1")
model = CGAN(device = device)

tmp_folder = "tmp"
os.makedirs(tmp_folder, exist_ok= True)

train_data= MNIST("../../Data/", train = True, download=True, transform = T.ToTensor())
train_loader = DataLoader(train_data, batch_size= 256)

optimizer_D = Adam(model.D.parameters(), lr = 2e-4, betas = (0.5, 0.999))
optimizer_G = Adam(model.G.parameters(), lr = 2e-4, betas = (0.5, 0.999))

pbar = tqdm(range(n_epoch))
loss_plot = {}
fixed_sample= torch.rand(100,100)
label_sample = torch.zeros((100,10))
for i in range(10):
    label_sample[i::10, i] = 1
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
        gen = model.generate(fixed_sample, label_sample).view(-1, 1, 28, 28)
        save_image(gen, f"{tmp_folder}/{ep}_save.png" ,nrow = 10)
for key in loss_plot.keys():
    plt.plot(loss_plot[key],label=key)
    plt.legend()
plt.savefig("./loss.png")