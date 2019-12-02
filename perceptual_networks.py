import os
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    '''
    The first layers of Torchvision's pretrained AlexNet
    Args:
        layer (int): The layer from which the features are extracted
        frozen (bool): Whether the network can be trained or not
        sigmoid_out: Whether to add a nn.Sigmoid layer to normalize output 
    '''
    def __init__(self, layer=5, pretrained=True, frozen=True, sigmoid_out=True):
        super(AlexNet, self).__init__()
        self.layer = layer
        self.pretrained = pretrained
        self.frozen = frozen
        self.sigmoid_out = sigmoid_out

        os.environ['TORCH_HOME'] = './'
        original_model = models.alexnet(pretrained=pretrained)
        self.features = nn.Sequential(
            *list(original_model.features.children())[:layer]
        )
        if sigmoid_out:
            self.features.add_module('sigmoid',nn.Sigmoid())
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    
    def __str__(self):
        return (
            f'AlexNet(layer={self.layer}, pretrained={self.pretrained}, '
            f'frozen={self.frozen}, sigmoid_out={self.sigmoid_out})'
        )