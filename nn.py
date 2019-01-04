import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def clip_weights(net, value=0.01):
    for p in net.parameters():
        p.data.clamp_(-value, value)

def gradient_penalty(f, device, real, fake=None):
    def interpolate(a, b=None):
        if type(b) is str and b == 'dragan':
            # interpolation in DRAGAN
            b = a + 0.5 * a.std() * torch.rand_like(a).to(device)
        if type(b) is str and b == 'lsgan-gp':
            # interpolation in LSGAN-GP (improved LSGAN)
            b = a + 30 * torch.rand_like(a).to(device)
        alpha = torch.rand(a.size(0), 1, 1, 1).to(device)
        inter = a + alpha * (b - a)
        return inter
    x = interpolate(real, fake).requires_grad_(True)
    pred = f(x)
    if isinstance(pred, tuple):
        pred = pred[0]
    grad = autograd.grad(
        outputs=pred, inputs=x,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.size(0), -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp

class Generator_32(nn.Module):
    def __init__(self):
        super(Generator_32, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, bias=False), 
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

class Discriminator_32(nn.Module):
    def __init__(self):
        super(Discriminator_32, self).__init__()
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
            nn.Conv2d(256, 1, 4, 1, bias=False), 
        )
    def forward(self, x):
        return self.layers(x).view(-1, )

class Generator_64(nn.Module):
    def __init__(self):
        super(Generator_64, self).__init__()
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

class Discriminator_64(nn.Module):
    def __init__(self):
        super(Discriminator_64, self).__init__()
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
        
        if args.img_size == 32:
            self.netG = Generator_32()
            self.netD = Discriminator_32()
        if args.img_size == 64:
            self.netG = Generator_64()
            self.netD = Discriminator_64()
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.optimG = optim.Adam(self.netG.parameters(), lr=args.g_lr, betas=args.g_betas)
        self.optimD = optim.Adam(self.netD.parameters(), lr=args.d_lr, betas=args.d_betas)
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
        if self.mode == 'lsgan':
            # g_loss = 0.5 * F.mse_loss(d_fake, torch.ones_like(d_fake).to(self.device))
            g_loss = F.mse_loss(d_fake, torch.ones_like(d_fake).to(self.device))
            errG['g_loss'] = g_loss.item()
        if self.mode == 'wgan-gp':
            g_loss = -d_fake.mean()
            errG['g_loss'] = g_loss.item()
        if self.mode == 'lsgan-gp':
            g_loss = F.mse_loss(d_fake, torch.ones_like(d_fake).to(self.device))
            errG['g_loss'] = g_loss.item()
        if self.mode == 'dragan':
            g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake).to(self.device))
            errG['g_loss'] = g_loss.item()
        if self.mode == 'gan-qp-l1':
            g_loss = (d_real - d_fake).mean()
            errG['g_loss'] = g_loss.item()
        if self.mode == 'gan-qp-l2':
            g_loss = (d_real - d_fake).mean()
            errG['g_loss'] = g_loss.item()
        
        self.optimG.zero_grad()
        g_loss.backward()
        self.optimG.step()
        return errG
    
    def trainD(self, x_real, z):
        if self.mode == 'wgan':
            clip_weights(self.netD, 0.01)
        
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
        if self.mode == 'lsgan':
            d_loss_real = F.mse_loss(d_real, torch.ones_like(d_real).to(self.device))
            d_loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake).to(self.device))
            # d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss = d_loss_real + d_loss_fake
            errD['d_loss_real'] = d_loss_real.item()
            errD['d_loss_fake'] = d_loss_fake.item()
            errD['d_loss'] = d_loss.item()
        if self.mode == 'wgan-gp':
            d_loss_real = d_real.mean()
            d_loss_fake = d_fake.mean()
            wd = d_loss_real - d_loss_fake
            d_loss = -wd
            d_gp = gradient_penalty(self.netD, self.device, x_real, x_fake)
            d_loss = d_loss + 10 * d_gp
            errD['d_loss_real'] = d_loss_real.item()
            errD['d_loss_fake'] = d_loss_fake.item()
            errD['d_gp'] = d_loss.item()
            errD['d_loss'] = d_loss.item()
        if self.mode == 'lsgan-gp':
            d_loss_real = F.mse_loss(d_real, torch.ones_like(d_real).to(self.device))
            d_loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake).to(self.device))
            d_gp = gradient_penalty(self.netD, self.device, x_real, 'lsgan-gp')
            d_loss = d_loss_real + d_loss_fake + 150 * d_gp
            errD['d_loss_real'] = d_loss_real.item()
            errD['d_loss_fake'] = d_loss_fake.item()
            errD['d_gp'] = d_loss.item()
            errD['d_loss'] = d_loss.item()
        if self.mode == 'dragan':
            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real).to(self.device))
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake).to(self.device))
            d_gp = gradient_penalty(self.netD, self.device, x_real, 'dragan')
            d_loss = d_loss_real + d_loss_fake
            errD['d_loss_real'] = d_loss_real.item()
            errD['d_loss_fake'] = d_loss_fake.item()
            errD['d_gp'] = d_loss.item()
            errD['d_loss'] = d_loss.item()
        if self.mode == 'gan-qp-l1':
            d_loss = d_real - d_fake
            d_norm = 10 * (x_real - x_fake).abs().view(x_real.size(0), -1).mean(dim=1, keepdim=True)
            d_loss = (-d_loss_ + 0.5 * d_loss_**2 / d_norm).mean()
            errD['d_loss_'] = d_loss_.mean().item()
            errD['d_norm'] = d_norm.item()
            errD['d_loss'] = d_loss.item()
        if self.mode == 'gan-qp-l2':
            d_loss_ = d_real - d_fake
            d_norm = 10 * (x_real - x_fake).pow(2).view(x_real.size(0), -1).mean(dim=1, keepdim=True).sqrt()
            d_loss = (-d_loss_ + 0.5 * d_loss_**2 / d_norm).mean()
            errD['d_loss_'] = d_loss_.mean().item()
            errD['d_norm'] = d_norm.item()
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