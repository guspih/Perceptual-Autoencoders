import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt

def run_epoch(model, dataloader, loss, optimizer,
    epoch_name='Epoch', train=True
):
    '''
    Trains a given model for one epoch
    Will automatically move data to gpu if model is on the gpu
    Args:
        model (nn.Module): The network to be trained
        dataloader (data.DataLoader): Torch DataLoader to load epoch data
        loss (f(output, target)->[tensor]): Loss calculation function
        optimizer (optim.Optimizer): Optimizer for use in training
        epoch_name (str): Name of the epoch (usually a number)
        train (bool): Whether to run this epoch to train or just to evaluate
    Returns: ([float]) The mean batch losses of the epoch
    '''
    start_time = time.time()
    gpu = next(model.parameters()).is_cuda

    if train:
        model.train()
    else:
        model.eval()
    epoch_losses = []
    for batch_id, (batch_data, batch_labels) in enumerate(dataloader):
        if gpu:
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()
        optimizer.zero_grad()
        output = model(batch_data)
        losses = loss(output, batch_labels)
        if batch_id == 0:
            epoch_losses = [
                loss.item() for loss in losses
            ]
        else:
            epoch_losses = [
                epoch_losses[i] + losses[i].item() for i in range(len(losses))
            ]
        losses[0].backward()
        if train:
            optimizer.step()
        print(
            '\r{} - [{}/{}] - Losses: {}, Time elapsed: {}s'.format(
                epoch_name, batch_id+1, len(dataloader),
                ', '.join(
                    ['{0:.5f}'.format(l/(batch_id+1)) for l in epoch_losses]
                ),
                '{0:.1f}'.format(time.time()-start_time)
            ),end=''
        )

    return [l/(batch_id+1) for l in epoch_losses]

def run_training(model, train_loader, val_loader, loss,
    optimizer, save_path, epochs, epoch_update=None
):
    '''
    Args:
        model (nn.Module): The network to be trained
        train_loader (data.Dataloader): Dataloader for training data
        val_loader (data.Dataloader): Dataloader for validation data
        loss (f(output, target)->[tensor]): Loss calculation function
        optimizer (optim.Optimizer): Optimizer for use in training
        save_path (str): Path to folder where the model will be stored
        epochs (int): Number of epochs to train for
        epoch_update (f(epoch, train_loss, val_loss) -> bool): Function to run
            at the end of a epoch. Returns whether to early stop
    Returns (nn.Module, str, float, int): The model, path, val loss, and epochs
    '''
    save_file = (
        model. __class__.__name__ + 
        datetime.datetime.now().strftime('_%Y-%m-%d_%Hh%Mm%Ss.pt')
    )
    if save_path != '':
        save_file = save_path + '/' + save_file

    torch_model_save(model, save_file)
    best_validation_loss = float('inf')
    best_epoch = 0
    for epoch in range(1,epochs+1):
        training_losses = run_epoch(
            model, train_loader, loss, optimizer,
            'Train {}'.format(epoch), train=True
        )

        validation_losses = run_epoch(
            model, val_loader, loss, optimizer,
            'Validation {}'.format(epoch), train=False
        )
        
        print(
            f'\rEpoch {epoch} - '
            f'Train loss {training_losses[0]:.5f} - '
            f'Validation loss {validation_losses[0]:.5f}',
            ' '*35
        )

        if validation_losses[0] < best_validation_loss:
            torch_model_save(model, save_file)
            best_validation_loss = validation_losses[0]
            best_epoch = epoch
        
        if not epoch_update is None:
            early_stop = epoch_update(epoch, training_losses, validation_losses)
            if early_stop:
                break

    model = torch.load(save_file)
    return model, save_file, best_validation_loss, best_epoch

class EarlyStopper():
    '''
    An implementation of Early stopping for run_training
    Args:
        patience (int): How many epochs without progress until stopping early
    '''
    
    def __init__(self, patience=20):
        self.patience = patience
        self.current_patience = patience
        self.best_loss = 99999999999999
    
    def __call__(self, epoch, train_losses, val_losses):
        if val_losses[0] < self.best_loss:
            self.best_loss = val_losses[0]
            self.current_patience = self.patience
        else:
            self.current_patience -= 1
            if self.current_patience == 0:
                return True
        return False

def fc_net(input_size, layers, activation_functions):
    '''
    Creates a simple fully connected network
    Args:
        input_size (int): Input size to the network
        layers ([int]): Layer sizes
        activation_functions ([f()->nn.Module]): class of activation functions
    Returns: (nn.Sequential)
    '''
    if not isinstance(activation_functions, list):
        activation_functions = [
            activation_functions for _ in range(len(layers)+1)
        ]

    network = nn.Sequential()
    layers.insert(0,input_size)
    for layer_id in range(len(layers)-1):
        network.add_module(
            'linear{}'.format(layer_id),
            nn.Linear(layers[layer_id], layers[layer_id+1])
        )
        if not activation_functions[layer_id] is None:
            network.add_module(
                'activation{}'.format(layer_id),
                activation_functions[layer_id]()
            )
    return network

def torch_model_save(model, file_path):
    '''
    Saves a cpu version of the given model at file_path
    Args:
        model (nn.Module): Model to save
        file_path (str): Path to file to store the model in
    '''
    device = next(model.parameters()).device
    model.cpu()
    torch.save(model, file_path)
    model.to(device)
