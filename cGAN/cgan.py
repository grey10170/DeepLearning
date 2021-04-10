import torch.nn as nn
import torch

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x: torch.Tensor):
        return x.view(self.shape)

class CGAN(nn.Module):
    def __init__(self, input_size = (28,28), latent_dim = 100, label_dim = 10 ,device = None):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(input_size[0]*input_size[1]+ label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(), # 0 to 1
        )
        self.G = nn.Sequential(
            nn.Linear(latent_dim+ label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024 , input_size[0]*input_size[1]),
            nn.Tanh() # -1 to 1
        )
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.device = device
        if self.device is not None:
            self.D.to(self.device)
            self.G.to(self.device)
    def train(self, train_loader, optimizer_D, optimizer_G):
        #training of 1 epoch
        #train_loader will give (b,1,28,28)
        loss_dict= {}
        for key in ["loss_D_fake", "loss_D_real", "loss_G"]:
            loss_dict[key] = 0
        for idx, (img, label ) in enumerate(train_loader):
            batch_size = img.shape[0]
            zeros = torch.zeros((batch_size, self.label_dim))
            zeros[range(batch_size), label] = 1
            label = zeros
            label = label.to(self.device).view(batch_size, -1)
            for _ in range(1):
                optimizer_D.zero_grad()
                img = img.view(batch_size, -1)
                real_img = img.to(self.device)
                real_pred = self.D(torch.cat([real_img, label], dim = 1))
                fake_latent = torch.rand((batch_size, self.latent_dim)).to(self.device)
                fake_img = self.G(torch.cat([fake_latent, label], dim = 1))
                fake_pred = self.D(torch.cat([fake_img, label], dim = 1 ))
                loss_D_fake = torch.mean(-torch.log(1-fake_pred))
                loss_D_real = torch.mean(-torch.log(real_pred))
                loss_D = (loss_D_fake + loss_D_real)
                loss_D.backward()
                loss_dict["loss_D_fake"] += loss_D_fake.item()
                loss_dict["loss_D_real"] += loss_D_real.item()
                optimizer_D.step()

            optimizer_G.zero_grad()
            fake_latent = torch.rand((batch_size, self.latent_dim)).to(self.device)
            fake_img = self.G(torch.cat([fake_latent, label],dim = 1))
            fake_pred = self.D(torch.cat([fake_img,label],dim = 1))
            # loss_G= torch.mean(torch.log(1-fake_pred))
            loss_G = torch.mean(-torch.log(fake_pred))
            loss_G.backward()
            loss_dict["loss_G"] += loss_G.item()
            optimizer_G.step()
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key]/ len(train_loader)
        return loss_dict
    @torch.no_grad()
    def generate(self, latent_code, label_code):
        code = torch.cat([latent_code, label_code], dim = 1).to(self.device)
        gen = self.G(code)
        return gen.detach().cpu()
            