import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchvision
import os

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from tqdm import *

from vae_model import VAE

torch.manual_seed(42) 
sns.set_style('dark')

def criterion(x_out, x_in, z_mu, z_logvar):
    bce_loss = F.binary_cross_entropy(x_out, x_in, size_average=False)
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (bce_loss + kld_loss) / x_out.size(0) # normalize by batch size
    return loss

# Train the model
def train(model, optimizer, dataloader, epochs=100):
    losses = []
    for _ in tqdm(range(epochs), desc='Epochs'):
        l = 0
        cnt = 0
        for images, _ in dataloader:
            x_in = Variable(images)
            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss = criterion(x_out, x_in, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            l += loss.data
            # losses.append(loss.data)
            cnt += 1
        losses.append(l / cnt)
    return losses

# Test the model
def test(model, dataloader):
    running_loss = 0.0
    for images, _ in dataloader:
        x_in = Variable(images)
        x_out, z_mu, z_logvar = model(x_in)
        loss = criterion(x_out, x_in, z_mu, z_logvar)
        running_loss = running_loss + (loss.data * x_in.size(0))
    return running_loss / len(dataloader.dataset)

def visualize_loss(loss, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(f'sample/{epoch}_epoch_loss.png', dpi = 300)
    plt.show()

# Save the results
def visualize_mnist_vae_generation(model, dataloader, index, num=16, sample_dir="./sample"):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
    
    LQ_path = os.path.join(sample_dir, "LQ_generation")
    HQ_path = os.path.join(sample_dir, "HQ_generation")
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    if not os.path.exists(LQ_path):
        os.mkdir(LQ_path)
    if not os.path.exists(HQ_path):
        os.mkdir(HQ_path)

    images, _ = next(iter(dataloader))
    images = images[0:num,:,:]
    x_in = Variable(images)
    noise = torch.randn_like(x_in)
    # x_out, _, _ = model(x_in)
    x_out, _, _ = model(noise)
    x_out = x_out.data


    grid_LQ = make_grid(x_out, nrow=4)
    img_LQ = torchvision.transforms.ToPILImage()(grid_LQ)
    img_LQ.save(os.path.join(LQ_path, f"{index}.png"))
    img_LQ.close()

    grid_HQ = make_grid(images, nrow=4)
    img_HQ = torchvision.transforms.ToPILImage()(grid_HQ)
    img_HQ.save(os.path.join(HQ_path, f"{index}.png"))
    img_HQ.close()
    # torchvision.utils.save_image(images, os.path.join(HQ_path, f"{index}.png"))
    # torchvision.utils.save_image(x_out, os.path.join(LQ_path, f"{index}.png"))
    # imshow(make_image_grid(images))
    # imshow(make_image_grid(x_out))

def visualize_mnist_vae_reconstruction(model, dataloader, index, num=16, sample_dir="./sample"):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
    
    LQ_path = os.path.join(sample_dir, "LQ_reconstruction")
    HQ_path = os.path.join(sample_dir, "HQ_reconstruction")
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    if not os.path.exists(LQ_path):
        os.mkdir(LQ_path)
    if not os.path.exists(HQ_path):
        os.mkdir(HQ_path)

    images, _ = next(iter(dataloader))
    images = images[0:num,:,:]
    x_in = Variable(images)
    # noise = torch.randn_like(x_in)
    x_out, _, _ = model(x_in)
    # x_out, _, _ = model(noise)
    x_out = x_out.data

    grid_LQ = make_grid(x_out, nrow=4)
    img_LQ = torchvision.transforms.ToPILImage()(grid_LQ)
    img_LQ.save(os.path.join(LQ_path, f"{index}.png"))
    img_LQ.close()

    grid_HQ = make_grid(images, nrow=4)
    img_HQ = torchvision.transforms.ToPILImage()(grid_HQ)
    img_HQ.save(os.path.join(HQ_path, f"{index}.png"))
    img_HQ.close()
    # imshow(make_image_grid(images))
    # imshow(make_image_grid(x_out))


def main():
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters())
    # Data loaders
    trainloader = DataLoader(MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    testloader = DataLoader(MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    
    epoch = 200
    train_losses = train(model, optimizer, trainloader, epoch)
    visualize_loss(train_losses, epoch)

    # Testing
    # test_loss = test(model,testloader)
    # print(test_loss)
    visualize_mnist_vae_generation(model, testloader, epoch)
    visualize_mnist_vae_reconstruction(model, testloader, epoch)

if __name__ == '__main__':
    main()