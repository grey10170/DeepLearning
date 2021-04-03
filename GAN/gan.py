import torch.nn as nn
import torch

class GAN(nn.Module):
    def __init__(self, input_size = (28,28), latent_dim = 10 ,device = None):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(input_size[0]*input_size[1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(), # 0 to 1
        )
        self.G = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512 , input_size[0]*input_size[1]),
            nn.Tanh() # -1 to 1
        )
        self.latent_dim = latent_dim
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
        for idx, (img, _ ) in enumerate(train_loader):
            batch_size = img.shape[0]

            optimizer_D.zero_grad()
            img = img.view(batch_size, -1)
            real_img = img.to(self.device)
            real_pred = self.D(real_img)
            fake_latent = torch.rand((batch_size, self.latent_dim)).to(self.device)
            fake_img = self.G(fake_latent)
            fake_pred = self.D(fake_img)
            loss_D_fake = torch.mean(-torch.log(1-fake_pred))
            loss_D_real = torch.mean(-torch.log(real_pred))
            loss_D = (loss_D_fake + loss_D_real)/2
            loss_D.backward()
            loss_dict["loss_D_fake"] += loss_D_fake.item()
            loss_dict["loss_D_real"] += loss_D_real.item()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_latent = torch.rand((batch_size, self.latent_dim)).to(self.device)
            fake_img = self.G(fake_latent)
            fake_pred = self.D(fake_img)
            loss_G= torch.mean(torch.log(1-fake_pred))
            # loss_G = torch.mean(-torch.log(fake_pred))
            loss_G.backward()
            loss_dict["loss_G"] += loss_G.item()
            optimizer_G.step()
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key]/ len(train_loader)
        return loss_dict
    @torch.no_grad()
    def example(self, latent_code):
        code = latent_code.to(self.device)
        gen = self.G(code)
        return gen.detach().cpu()
            