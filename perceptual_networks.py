import os
import torch.nn as nn
import torchvision.models as models

# Dictionary off torchvision models and the attribute 'paths' to their features
architecture_features = {
    'alexnet' : ['features'],
    'vgg11' : ['features'],
    'vgg11_bn' : ['features'],
    'vgg13' : ['features'],
    'vgg13_bn' : ['features'],
    'vgg16' : ['features'],
    'vgg16_bn' : ['features'],
    'vgg19' : ['features'],
    'vgg19_bn' : ['features'],
    'densenet121' : ['features'],
    'densenet161' : ['features'],
    'densenet169' : ['features'],
    'densenet201' : ['features'],
    'resnet18' : [],
    'resnet34' : [],
    'resnet50' : [],
    'resnet101' : [],
    'resnet152' : [],
    'wide_resnet50_2' : [],
    'wide_resnet101_2' : [],
    'shufflenet_v2_x1_0' : [],
    'shufflenet_v2_x2_0' : [],
    'mobilenet_v2' : ['features'],
    'googlenet' : [],
    'inception_v3' : [],
    'squeezenet1_0' : ['features'],
    'squeezenet1_1' : ['features']
}

def AlexNet(layer=5, pretrained=True, frozen=True, sigmoid_out=True):
    return SimpleExtractor('alexnet',layer,frozen,sigmoid_out)

class SimpleExtractor(nn.Module):
    '''
    A simple feature extractor for torchvision models
    Args:
        architecture (str): The architecture to extract from
        layer (int): The sub-module in 'features' to extract at
        frozen (bool): Whether the network can be trained
        sigmoid_out (bool): Whether to normalize the output with a sigmoid
    '''
    def __init__(self, architecture, layer, frozen=True, sigmoid_out=True):
        super(SimpleExtractor, self).__init__()
        self.architecture = architecture
        self.layer = layer
        self.frozen = frozen
        self.sigmoid_out = sigmoid_out
    
        os.environ['TORCH_HOME'] = './'
        original_model = models.__dict__[architecture](pretrained=True)
        original_features = original_model
        for attribute in architecture_features[architecture]:
            original_features = getattr(original_features, attribute)
        self.features = nn.Sequential(
            *list(original_features.children())[:layer]
        )
        if sigmoid_out:
            self.features.add_module('sigmoid',nn.Sigmoid())
        if frozen:
            self.eval()
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def __str__(self):
        return (
            f'{self.architecture}(layer={self.layer}, '
            f'frozen={self.frozen}, sigmoid_out={self.sigmoid_out})'
        )
