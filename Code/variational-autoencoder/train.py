import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import os

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid as make_image_grid
from tqdm import tnrange

from vae_model import VAE

torch.manual_seed(2017) 
sns.set_style('dark')

def criterion(x_out, x_in, z_mu, z_logvar):
    bce_loss = F.binary_cross_entropy(x_out,x_in,size_average=False)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (bce_loss + kld_loss) / x_out.size(0) # normalize by batch size
    return loss

# Train
def train(model,optimizer,dataloader,epochs=15):
    losses = []
    for epoch in tnrange(epochs,desc='Epochs'):
        for images,_ in dataloader:
            x_in = Variable(images)
            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss = criterion(x_out,x_in,z_mu,z_logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
    return losses

# Test
def test(model, dataloader):
    running_loss = 0.0
    for images, _ in dataloader:
        x_in = Variable(images)
        x_out, z_mu, z_logvar = model(x_in)
        loss = criterion(x_out, x_in, z_mu, z_logvar)
        running_loss = running_loss + (loss.data * x_in.size(0))
    return running_loss/len(dataloader.dataset)

def visualize_mnist_vae(model, dataloader, num=16, sample_dir="./sample"):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
    
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    images, _ = next(iter(dataloader))
    images = images[0:num,:,:]
    x_in = Variable(images)
    x_out, _, _ = model(x_in)
    x_out = x_out.data
    torchvision.utils.save_image(images, os.path.join(sample_dir, "input.png"))
    torchvision.utils.save_image(x_out, os.path.join(sample_dir, "output.png"))
    imshow(make_image_grid(images))
    imshow(make_image_grid(x_out))


def main():
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters())
    # Data loaders
    trainloader = DataLoader(
        MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True)
    testloader = DataLoader(
        MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True)
    
    train_losses = train(model,optimizer,trainloader)
    plt.figure(figsize=(10,5))
    plt.plot(train_losses)
    plt.show()

    # Testing
    # test_loss = test(model,testloader)
    # print(test_loss)

    visualize_mnist_vae(model,testloader)

if __name__ == '__main__':
    main()