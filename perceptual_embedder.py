# Library imports
import torch
import numpy as np
import torchvision.models as models
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# File imports
from utility import run_training, EarlyStopper
from VAE import _create_coder, TemplateVAE

class PerceptualEmbedder(TemplateVAE):
    '''
    A Convolutional Variational Embedder for images trained to recreate embeddings
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use
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
        assert perceptual_net != None, \
            'For PerceptualEmbedder, perceptual_net cannot be None'

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net
        
        inp = torch.rand((1,3,input_size[0],input_size[1]))
        out = self.perceptual_net(inp.to(next(perceptual_net.parameters()).device))
        self.perceptual_size = out.numel()
        self.perceptual_loss = True

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
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dimensions, deconv_flat_size),
            nn.ReLU(),
            nn.Linear(deconv_flat_size, int(self.perceptual_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.perceptual_size/2), self.perceptual_size)
        )

    def loss(self, output, x):
        rec_y, z, mu, logvar = output
        
        y = self.perceptual_net(x)
        REC = F.mse_loss(rec_y, y, reduction='mean')

        if self.variational:
            KLD = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return REC + self.gamma*KLD, REC, KLD
        else:
            return [REC]

class PerceptualPreEmbedder(TemplateVAE):
    '''
    An fc autoencoder that encodes the features of a perceptual network
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use
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
        assert perceptual_net != None, \
            'For PerceptualPreEmbedder, perceptual_net cannot be None'

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net
        
        inp = torch.rand((1,3,input_size[0],input_size[1]))
        out = self.perceptual_net(inp.to(next(perceptual_net.parameters()).device))
        self.perceptual_size = out.numel()
        self.perceptual_loss = True

        self.encoder = nn.Sequential(
            nn.Linear(self.perceptual_size, int(self.perceptual_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.perceptual_size/2), 1024),
            nn.ReLU()
        )

        self.mu = nn.Linear(1024, self.z_dimensions)
        self.logvar = nn.Linear(1024, self.z_dimensions)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dimensions, 1024),
            nn.ReLU(),
            nn.Linear(1024, int(self.perceptual_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.perceptual_size/2), self.perceptual_size)
        )

    def encode(self, x):
        y = self.perceptual_net(x)
        y = y.view(y.size(0),-1)
        y = self.encoder(y)
        mu = self.mu(y)
        logvar = self.logvar(y)
        return mu, logvar

    def loss(self, output, x):
        rec_y, z, mu, logvar = output
        
        y = self.perceptual_net(x)
        y = y.view(y.size(0),-1)

        REC = F.mse_loss(rec_y, y, reduction='mean')

        if self.variational:
            KLD = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return REC + self.gamma*KLD, REC, KLD
        else:
            return [REC]

class PerceptualReconstructer(TemplateVAE):
    '''
    A CVAE that encodes perceptual features and reconstructs the images
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
        assert perceptual_net != None, \
            'For PerceptualReconstructer, perceptual_net cannot be None'

        #Attributes
        self.input_size = input_size
        self.z_dimensions = z_dimensions
        self.variational = variational
        self.gamma = gamma
        self.perceptual_net = perceptual_net

        inp = torch.rand((1,3,input_size[0],input_size[1]))
        out = self.perceptual_net(inp.to(next(perceptual_net.parameters()).device))
        self.perceptual_size = out.numel()
        self.perceptual_loss = True

        self.encoder = nn.Sequential(
            nn.Linear(self.perceptual_size, int(self.perceptual_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.perceptual_size/2), 1024),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(1024, self.z_dimensions)
        self.logvar = nn.Linear(1024, self.z_dimensions)

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

    def encode(self, x):
        y = self.perceptual_net(x)
        y = y.view(y.size(0),-1)
        y = self.encoder(y)
        mu = self.mu(y)
        logvar = self.logvar(y)
        return mu, logvar

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