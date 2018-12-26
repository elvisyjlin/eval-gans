import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(64, 64, 3, 1, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False), 
            nn.Tanh(), 
        )
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.layers(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(64, 64, 3, 1, padding=1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(512, 1, 4, 1, bias=False), 
        )
    def forward(self, x):
        return self.layers(x).view(-1, )

class GAN():
    def __init__(self, args):
        self.device = args.device
        self.mode = args.mode
        
        self.netG = Generator()
        self.netG.to(self.device)
        self.netD = Discriminator()
        self.netD.to(self.device)
        self.optimG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=args.betas)
        self.optimD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=args.betas)
        self.vardict = {
            'netG': self.netG, 
            'netD': self.netD, 
            'optimG': self.optimG, 
            'optimD': self.optimD
        }
        
        print('netG')
        summary(self.netG, (args.z_dim, ), use_gpu=True)
        print('netD')
        summary(self.netD, (3, args.img_size, args.img_size), use_gpu=True)
    
    def init(self, init_fn):
        self.netG.apply(init_fn)
        self.netD.apply(init_fn)
    
    def trainG(self, x_real, z):
        x_fake = self.netG(z)
        d_real = self.netD(x_real)
        d_fake = self.netD(x_fake)
        
        errG = {}
        if self.mode == 'dcgan':
            g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake).to(self.device))
            errG['g_loss'] = g_loss.item()
        if self.mode == 'wgan':
            g_loss = -d_fake.mean()
            errG['g_loss'] = g_loss.item()
        
        self.optimG.zero_grad()
        g_loss.backward()
        self.optimG.step()
        return errG
    
    def trainD(self, x_real, z):
        x_fake = self.netG(z).detach()
        d_real = self.netD(x_real)
        d_fake = self.netD(x_fake)
        
        errD = {}
        if self.mode == 'dcgan':
            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real).to(self.device))
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake).to(self.device))
            d_loss = d_loss_real + d_loss_fake
            errD['d_loss_real'] = d_loss_real.item()
            errD['d_loss_fake'] = d_loss_fake.item()
            errD['d_loss'] = d_loss.item()
        if self.mode == 'wgan':
            d_loss_real = d_real.mean()
            d_loss_fake = d_fake.mean()
            wd = d_loss_real - d_loss_fake
            d_loss = -wd
            errD['d_loss_real'] = d_loss_real.item()
            errD['d_loss_fake'] = d_loss_fake.item()
            errD['d_loss'] = d_loss.item()
        
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        return errD
    
    def save(self, file, options=['netG', 'netD', 'optimG', 'optimD']):
        states = {}
        for name in options:
            states[name] = self.vardict[name].state_dict()
        torch.save(states, file)
    
    def load(self, file):
        states = torch.load(file)
        for name in states:
            self.vardict[name].load_state_dict(states[name])
    
    def train(self):
        self.netG.train()
        self.netD.train()
    
    def eval(self):
        self.netG.eval()
        self.netD.eval()