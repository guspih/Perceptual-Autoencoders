# Library imports
import random
import torch
import numpy as np
import torchvision.models as models
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import datetime
import time
import sys
import os
import matplotlib.pyplot as plt

# File imports
from utility import run_training, EarlyStopper

def _create_coder(channels, kernel_sizes, strides, conv_types,
    activation_types, paddings=(0,0), batch_norms=False
):
    '''
    Function that creates en- or decoders based on parameters
    Args:
        channels ([int]): Channel sizes per layer. 1 more than layers
        kernel_sizes ([int]): Kernel sizes per layer
        strides ([int]): Strides per layer
        conv_types ([f()->type]): Type of the convoultion module per layer
        activation_types ([f()->type]): Type of activation function per layer
        paddings ([(int, int)]): The padding per layer
        batch_norms ([bool]): Whether to use batchnorm on each layer
    Returns (nn.Sequential): The created coder
    '''
    if not isinstance(conv_types, list):
        conv_types = [conv_types for _ in range(len(kernel_sizes))]

    if not isinstance(activation_types, list):
        activation_types = [activation_types for _ in range(len(kernel_sizes))]

    if not isinstance(paddings, list):
        paddings = [paddings for _ in range(len(kernel_sizes))]
        
    if not isinstance(batch_norms, list):
        batch_norms = [batch_norms for _ in range(len(kernel_sizes))]

    coder = nn.Sequential()
    for layer in range(len(channels)-1):
        coder.add_module(
            'conv'+ str(layer), 
            conv_types[layer](
                in_channels=channels[layer], 
                out_channels=channels[layer+1],
                kernel_size=kernel_sizes[layer],
                stride=strides[layer]
            )
        )
        if batch_norms[layer]:
            coder.add_module(
                'norm'+str(layer),
                nn.BatchNorm2d(channels[layer+1])
            )
        if not activation_types[layer] is None:
            coder.add_module('acti'+str(layer),activation_types[layer]())

    return coder

class TemplateVAE(nn.Module):
    '''
    A template class for Variational Autoencoders to minimize code duplication
    Args:
        input_size (int,int): The height and width of the input image
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''
    
    def __str__(self):
        string = super().__str__()[:-1]
        string = string + '  (variational): {}\n  (gamma): {}\n)'.format(
                self.variational,self.gamma
            )
        return string

    def __repr__(self):
        string = super().__repr__()[:-1]
        string = string + '  (variational): {}\n  (gamma): {}\n)'.format(
                self.variational,self.gamma
            )
        return string
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        out = eps.mul(std).add_(mu)
        return out

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational:
            z = self.sample(mu, logvar)
        else:
            z = mu
        rec_x = self.decode(z)
        return rec_x, z, mu, logvar

    def loss(self, output, x):
        rec_x, z, mu, logvar = output
        if self.perceptual_loss:
            x = self.perceptual_net(x)
            rec_x = self.perceptual_net(rec_x)
        else:
            x = x.reshape(x.size(0), -1)
            rec_x = rec_x.view(x.size(0), -1)
        REC = F.mse_loss(rec_x, x, reduction='mean')

        if self.variational:
            KLD = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return REC + self.gamma*KLD, REC, KLD
        else:
            return [REC]

class FourLayerCVAE(TemplateVAE):
    '''
    A Convolutional Variational Autoencoder for images
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''

    def __init__(self, input_size=(64,64), z_dimensions=32,
        variational=True, gamma=20.0, perceptual_net=None
    ):
        super().__init__()

        #Parameter check
        if (input_size[0] - 64) % 16 != 0 or (input_size[1] - 64) % 16 != 0:
            raise ValueError(
                f'Input_size is {input_size}, but must be 64+16*N'
            )

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net

        self.perceptual_loss = not perceptual_net is None
            
        encoder_channels = [3,32,64,128,256]
        self.encoder = _create_coder(
            encoder_channels, [4,4,4,4], [2,2,2,2],
            nn.Conv2d, nn.ReLU,
            batch_norms=[True,True,True,True]
        )
        
        f = lambda x: np.floor((x - (2,2))/2)
        conv_sizes = f(f(f(f(np.array(input_size)))))
        conv_flat_size = int(encoder_channels[-1]*conv_sizes[0]*conv_sizes[1])
        self.mu = nn.Linear(conv_flat_size, self.z_dimensions)
        self.logvar = nn.Linear(conv_flat_size, self.z_dimensions)

        g = lambda x: int((x-64)/16)+1
        deconv_flat_size = g(input_size[0]) * g(input_size[1]) * 1024
        self.dense = nn.Linear(self.z_dimensions, deconv_flat_size)

        self.decoder = _create_coder(
            [1024,128,64,32,3], [5,5,6,6], [2,2,2,2],
            nn.ConvTranspose2d,
            [nn.ReLU,nn.ReLU,nn.ReLU,nn.Sigmoid],
            batch_norms=[True,True,True,False]
        )

        self.relu = nn.ReLU()

    def decode(self, z):
        y = self.dense(z)
        y = self.relu(y)
        y = y.view(
            y.size(0), 1024,
            int((self.input_size[0]-64)/16)+1,
            int((self.input_size[1]-64)/16)+1
        )
        y = self.decoder(y)
        return y

def show(imgs, block=False, save=None, heading='Figure', fig_axs=None, torchy=True):
    '''
    Paints a column of torch images
    Args:
        imgs ([3darray]): Array of images in shape (channels, width, height)
        block (bool): Whether the image should interupt program flow
        save (str / None): Path to save the image under. Will not save if None
        heading (str)): The heading to put on the image
        fig_axs (plt.Figure, axes.Axes): Figure and Axes to paint on
    Returns (plt.Figure, axes.Axes): The Figure and Axes that was painted
    '''
    if fig_axs is None:
        fig, axs = plt.subplots(1,len(imgs))
        if len(imgs) == 1:
            axs = [axs]
    else:
        fig, axs = fig_axs
        plt.figure(fig.number)
    fig.canvas.set_window_title(heading)
    for i, img in enumerate(imgs):
        if torchy:
            img = img[0].detach().permute(1,2,0)
        plt.axes(axs[i])
        plt.imshow(img)
    plt.show(block=block)
    plt.pause(0.001)
    if not save is None:
        plt.savefig(save)
    return fig, axs

def show_recreation(dataset, model, block=False, save=None):
    '''
    Shows a random image and the encoders attempted recreation
    Args:
        dataset (data.Dataset): Torch Dataset with the image data
        model (nn.Module): (V)AE model to be run
        block (bool): Whether to stop execution until user closes image
        save (str / None): Path to save the image under. Will not save if None
    '''
    with torch.no_grad():
        img1 = dataset[random.randint(0,len(dataset)-1)][0].unsqueeze(0)
        if next(model.parameters()).is_cuda:
            img1 = img1.cuda()
        img2, z, mu, logvar = model(img1)
    show(
        [img1.cpu(),img2.cpu()], block=block, save=save,
        heading='Random image recreation'
    )

def train_autoencoder(data, model, epochs, batch_size, gpu=False,
    display=False, save_path='checkpoints'
):
    '''
    Trains an autoencoder with the given data
    Args:
        data (tensor, tensor): Tuple with train and validation data
        model (nn.Module / str): Model or path to model to train
        epochs (int): Number of epochs to run
        batch_size (int): Size of batches
        gpu (bool): Whether to train on the GPU
        display (bool): Whether to display the recreated images
        save_path (str): Path to folder where the trained network will be stored
    Returns (nn.Module, str, float, int): The model, path, val loss, and epochs
    '''
    train_data, val_data = data
    train_data = TensorDataset(train_data, train_data)
    val_data = TensorDataset(val_data, val_data)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)

    if isinstance(model, str) and epochs != 0:
        model = torch.load(model, map_location='cpu')

    if gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    early_stop = EarlyStopper(patience=max(10, epochs/20))
    if display:
        epoch_update = lambda _a, _b, _c : show_recreation(
                train_data, model, block=False, save=save_path+'/image.png'
            ) or early_stop(_a,_b,_c)
    else:
        epoch_update = early_stop
    if epochs != 0:
        print(
            (
                'Starting autoencoder training. ' 
                f'Best checkpoint stored in ./{save_path}'
            )
        )   
        model, model_file, val_loss, actual_epochs = run_training(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            loss = model.loss,
            optimizer = optimizer,
            save_path = save_path,
            epochs = epochs,
            epoch_update = epoch_update
        )
    elif isinstance(model, str):
        model_file = model
    else:
        model_file = None

    if display:
        for batch_id in range(len(train_data)):
            show_recreation(train_data, model, block=True)
    
    return model, model_file, val_loss, actual_epochs

def encode_data(autoencoder, data, batch_size=512):
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    gpu = next(autoencoder.parameters()).is_cuda
    encoded_batches = []
    autoencoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = batch[0]
            if gpu:
                batch = batch.cuda()
            coded_batch = autoencoder.encode(batch)
            if gpu:
                coded_batch = (coded_batch[0].cpu(), coded_batch[1].cpu())
                batch = batch.cpu()
            encoded_batches.append(coded_batch[0])
    autoencoder.train()
    return torch.cat(encoded_batches, dim=0)