# Library imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import math
import datetime
import argparse
import os
import csv
import sys
from itertools import combinations_with_replacement, product

# File imports
from utility import run_training, run_epoch, fc_net, EarlyStopper
from VAE import FourLayerCVAE, train_autoencoder, encode_data
from perceptual_networks import AlexNet

# Dataset imports
from dataset_loader import split_data, load_lunarlander_data, \
    load_svhn_data, load_stl_data


def generate_autoencoders(index_file, dataset_name, data, epochs=100,
    batch_size=512, networks=[FourLayerCVAE], z_dims=[32,64,128],
    gammas=[0,0.001,0.01], perceptual_nets=[None, AlexNet()]
):
    '''
    Trains autoencoders with all combinations of the given parameters that are
    missing from index_file and adds them to index_file
    Args:
        index_file (str): Path to file to save model paths and parameters in
        dataset_name (str): Name of the dataset
        data (tensor, tensor): Tuple with train and validation data
        epochs (int): Maximum number of epochs to train each autoencoder for
        batch_size (int): Size of the batches
        networks ([f()->nn.Module]): Autoencoder implementations
        z_dims ([int]): The z_dim values to try
        gammas ([float]): The gamma values to try (0 = non-variational)
        perceptual_nets ([nn.Module/None]): Perceptual networks for loss
    '''

    #Create the index path + file if they don't exist already
    path = index_file.split(sep='/')[:-1]
    if len(path) > 0:
        try:
            os.makedirs('/'.join(path))
        except FileExistsError:
            pass
    if not os.path.isfile(index_file):
        with open(index_file, 'a') as index:
            index_writer = csv.writer(index, delimiter='\t')
            index_writer.writerow([
                'autoencoder_path',
                'dataset_name',
                'input_size',
                'epochs',
                'network',
                'z_dim',
                'gamma',
                'perceptual_net',
                'validation_loss'
            ])

    input_size = (data[0].size()[2], data[0].size()[3])

    # For each parameter combination
    for network,  z_dim,  gamma,  perceptual_net in product(
        networks, z_dims, gammas, perceptual_nets
    ):

        parameters = [
            dataset_name,
            str(input_size),
            str(epochs),
            str(network),
            str(z_dim),
            str(gamma),
            str(perceptual_net)
        ]
        
        # Check if a model with these parameters have already been trained
        already_trained = False
        with open(index_file, 'r') as index:
            index_reader = csv.reader(index, delimiter='\t')
            field_names = next(index_reader)
            for row in index_reader:
                if list(row[1:-1]) == parameters:
                    already_trained = True
                    break
        # If a model has already been trained a new one won't be
        if already_trained:
            continue

        # Initialize an autoencoder model with the given parameters
        model = network(
            input_size = input_size,
            z_dimensions = z_dim,
            variational = (gamma != 0),
            gamma = gamma,
            perceptual_net = perceptual_net
        )

        # Train the autoencoder with the data
        model, model_path, val_loss = train_autoencoder(
            data,
            model,
            epochs,
            batch_size,
            gpu=torch.cuda.is_available(),
            display=False,
            save_path='checkpoints'
        )

        # Save the path and parameters to index_file
        with open(index_file, 'a') as index:
            index_writer = csv.writer(index, delimiter='\t')
            index_writer.writerow([
                model_path,
                dataset_name,
                str(input_size),
                str(epochs),
                str(network),
                str(z_dim),
                str(gamma),
                str(perceptual_net),
                str(val_loss)
            ])

def generate_dense_architectures(hidden_sizes, hidden_nrs):
    '''
    Given acceptable sizes for hidden layers and acceptable number of layers,
    generates all feasible architectures to test.

    Args:
        hidden_sizes ([int]): List of acceptable sizes of the hidden layers
        hidden_nrs ([int]): List of acceptable number of layers
    
    Returns ([[int]]): List of architectures consisting of list of layer sizes
    '''
    archs = []
    hidden_sizes.sort(reverse=True)
    for hidden_nr in hidden_nrs:
        archs = archs + list(combinations_with_replacement(hidden_sizes, hidden_nr))
    return [list(arch) for arch in archs]

def run_experiment(results_file, dataset_name, train_data, validation_data,
    test_data, autoencoder_index, epochs, batch_size, predictor_architectures,
    predictor_hidden_functions, predictor_output_functions,
    allowed_ae_parameters={}
):
    '''
    Trains and tests fully connected networks with the given architectures on
    the given data, using autoencoders from autoencoder_index to encode the
    images. The results of the tests are saved to result_file
    Args:
        results_file (str): Path of the results file
        dataset_name (str): Name of the dataset (used to pick the correct AEs)
        train_data (tensor, tensor): Data and labels to train models on
        validation_data (tensor, tensor): Data and labels to validate models on
        test_data (tensor, tensor): Data and labels to test models on
        autoencoder_index (str): Path to index file of trained autoencoders
        epochs (int): Number of epochs to train each model for
        batch_size (int): Size of batches
        predictor_architectures ([[int]]): Architectures defined by layer sizes
        predictor_hidden_functions ([f()->nn.Module]): Hidden layer functions
        predictor_out_functions ([f()->nn.Module]): Output activation functions
        allowed_ae_parameters ({[any]}): Allowed parameters (all if empty)
    '''
    
    #Create the results path + file if they don't exist already
    path = results_file.split(sep='/')[:-1]
    if len(path) > 0:
        try:
            os.makedirs('/'.join(path))
        except FileExistsError:
            pass
    if not os.path.isfile(results_file):
        with open(results_file, 'a') as results:
            results_writer = csv.writer(results, delimiter='\t')
            results_writer.writerow([
                'autoencoder_path',
                'dataset_name',
                'input_size',
                'autoencoder_epochs',
                'autoencoder_network',
                'z_dim',
                'gamma',
                'perceptual_net',
                'autoencoder_val_loss',
                'predictor_path',
                'architecture',
                'hidden_function',
                'out_function',
                'predictor_epochs',
                'validation_MSE',
                'test_MSE',
                'Mean_L1_distance',
                'Mean_L2_distance',
                'Accuracy'
            ])
    # Setup variables, early stopping and losses that is used by all tests
    image_size = (train_data[0].size()[2], train_data[0].size()[3])
    label_size = train_data[1].size()[1]
    early_stop = EarlyStopper(patience=max(10, epochs/20))
    loss_function = torch.nn.MSELoss()
    losses = lambda output, target : [
        loss_function(output, target),
        torch.mean(torch.norm(output-target,1,dim=1)),
        torch.mean(torch.norm(output-target,2,dim=1)),
        torch.mean(
            torch.eq(torch.max(output,1)[1], torch.max(target,1)[1]).float()
        )
    ]

    # Collect paths and parameters of all autoencoders to use
    autoencoders = []
    with open(autoencoder_index, 'r') as index:
        index_reader = csv.reader(index, delimiter='\t')
        field_names = next(index_reader)
        for row in index_reader:
            if row[1] != dataset_name or row[2] != str(image_size):
                continue
            allowed_autoencoder = True
            for i, key in enumerate(field_names):
                if not key in allowed_ae_parameters:
                    continue
                if row[i] not in allowed_ae_parameters[key]:
                    allowed_autoencoder = False
                    break
            if allowed_autoencoder:
                autoencoders.append(row)
    
    # For all autoencoders run the test with all predictors
    for autoencoder_parameters in autoencoders:
        autoencoder_path = autoencoder_parameters[0]
        autoencoder = torch.load(autoencoder_path, map_location='cpu')
        
        # Encode and prepare the data
        train_encoded = encode_data(autoencoder,train_data[0],batch_size)
        train_dataset = TensorDataset(train_encoded, train_data[1])
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        
        val_encoded = encode_data(autoencoder,validation_data[0],batch_size)
        val_dataset = TensorDataset(val_encoded, validation_data[1])
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        
        test_encoded = encode_data(autoencoder,test_data[0],batch_size)
        test_dataset = TensorDataset(test_encoded, test_data[1])
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        encoding_size = train_encoded.size(-1)

        # Train and test all predictors on the given data
        for architecture, hidden_func, out_func in product(
            predictor_architectures,
            predictor_hidden_functions,
            predictor_output_functions
        ):
            # Initialize the predictor
            architecture = architecture.copy()
            architecture.append(label_size)
            act_functs = [hidden_func]*(len(architecture)-1) + [out_func]
            predictor = fc_net(
                input_size = encoding_size,
                layers = architecture,
                activation_functions = act_functs
            )
            optimizer = torch.optim.Adam(predictor.parameters())

            # Train the predictor
            predictor, predictor_path, validation_loss = run_training(
                predictor, train_loader, val_loader, losses,
                optimizer, 'checkpoints', epochs, epoch_update=early_stop
            )

            # Test the predictor
            test_losses = run_epoch(
                predictor, test_loader, losses, optimizer,
                epoch_name='Test',train=False
            )

            # Write the results to a .csv file
            with open(results_file, 'a') as results:
                results_writer = csv.writer(
                    results,
                    delimiter='\t',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL
                )
                results_writer.writerow(
                    autoencoder_parameters +
                    [
                        predictor_path, architecture, hidden_func,
                        out_func, epochs, validation_loss
                    ] +
                    test_losses
                )

def main():
    '''
    Given the autoencoder parameters and a dataset trains those autoencoders
    that are missing and then trains and tests the predictors specified by the
    predictor parameters for each autoencoer.
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        #To add a dataset, append its name here and preprocessing later
        '--data', type=str, choices=['lunarlander','stl10','svhn'],
        required=True, help='The dataset to test on'
    )
    parser.add_argument(
        '--ae_epochs', type=int, default=50,
        help='Nr of epochs to train autoencoders for'
    )
    parser.add_argument(
        '--ae_batch_size', type=int, default=512,
        help='Size of autoencoder batches'
    )
    parser.add_argument(
        #To add an autoencoder, append its name here and preprocessing later
        '--ae_networks', type=str, choices=['FourLayerCVAE'],
        default=['FourLayerCVAE'], nargs='+',
        help='The different autoencoder networks to use'
    )
    parser.add_argument(
        '--ae_zs', type=int, default=[64,128], nargs='+',
        help='The different autoencoder z_dims to use'
    )
    parser.add_argument(
        '--ae_gammas', type=float, default=[0,0.01], nargs='+',
        help='The different autoencoder gammas to use'
    )
    parser.add_argument(
        #To add a perceptual net, append its name here and preprocessing later
        '--ae_perceptuals', type=str, choices=['None', 'AlexNet'],
        default=['None', 'AlexNet'], nargs='+',
        help='The different autoencoder perceptual networks to use'
    )
    parser.add_argument(
        '--predictor_epochs', type=int, default=50,
        help='Nr of epochs to train predictors for'
    )
    parser.add_argument(
        '--predictor_batch_size', type=int, default=512,
        help='Size of predictor batches'
    )
    #TODO: Add arguments to use non-default architectures and functions
    parser.add_argument(
        '--autoencoder_index', type=str, default='autoencoder_index.csv',
        help='Path to store/load autoencoder paths/parameters to/from'

    )
    parser.add_argument(
        '--results_path', type=str, default='results.csv',
        help='Path to save results to'

    )
    #TODO: Make work
    parser.add_argument(
        '--no_gpu', action='store_true',
        help='The GPU will not be used even if it is available'
    )
    #TODO: Add verbosity control

    args = parser.parse_args()
    
    # Load autoencoder dataset, add code here to add new datasets
    if args.data == 'lunarlander':
        raise NotImplementedError(
            'Use gym_datagenerator.py to generate data '
            'then uncomment and add file names below'
        )
        #data, _ = load_lunarlander_data(
        #    './datasets/LunarLander-v2/<name_of_file>'
        #)
    elif args.data == 'stl10':
        data, _ = load_stl_data('./datasets/stl10/unlabeled_X.bin')
    elif args.data == 'svhn':
        data, _ = load_svhn_data('./datasets/svhn/extra_32x32.mat')
    else:
        raise ValueError(
            f'Dataset {args.data} does not match any implemented dataset name'
        )
    train_data, validation_data = split_data([data])
    train_data = train_data[0]
    validation_data = validation_data[0]

    # Get autoencoder networks, add code here to add new autoencoders
    networks = []
    for network in args.ae_networks:
        if network == 'FourLayerCVAE':
            networks.append(FourLayerCVAE)
        else:
            raise ValueError(
                f'{network} does not match any known autoencoder'
            )
    
    # Get perceptual networks, add code here to add new perceptual networks
    perceptual_nets = []
    for perceptual_net in args.ae_perceptuals:
        if perceptual_net == 'None':
            perceptual_nets.append(None)
        elif perceptual_net == 'AlexNet':
            perceptual_nets.append(AlexNet())
        else:
            raise ValueError(
                f'{perceptual_net} does not match any known perceptual net'
            )

    # Train the missing autoencoders
    generate_autoencoders(
        index_file = args.autoencoder_index,
        dataset_name = args.data,
        data = (train_data, validation_data), 
        epochs = args.ae_epochs,
        batch_size = args.ae_batch_size,
        networks = networks,
        z_dims = args.ae_zs,
        gammas = args.ae_gammas,
        perceptual_nets = perceptual_nets
    )

    # Load the predictor training and testing data, code here to add dataset
    if args.data == 'lunarlander':
        raise NotImplementedError(
            'Use gym_datagenerator.py to generate data '
            'then uncomment and add file names below'
        )
        #data, labels = load_lunarlander_data(
        #    './datasets/LunarLander-v2/<name_of_file>'
        #)
        #test_data, test_labels = load_lunarlander_data(
        #    './datasets/LunarLander-v2/<name_of_file>'
        #)
    elif args.data == 'stl10':
        data, labels = load_stl_data(
            './datasets/stl10/train_X.bin',
            './datasets/stl10/train_y.bin'
        )
        test_data, test_labels = load_stl_data(
            './datasets/stl10/test_X.bin',
            './datasets/stl10/test_y.bin'
        )
    elif args.data == 'svhn':
        data, labels = load_svhn_data(
            './datasets/svhn/train_32x32.mat'
        )
        test_data, test_labels = load_svhn_data(
            './datasets/svhn/test_32x32.mat'
        )
    else:
        raise ValueError(
            f'Dataset {args.data} does not match any implemented dataset name'
        )
    train_data, validation_data = split_data([data, labels])
    test_data = (test_data, test_labels)

    # Create architectures TODO: Add ability to control this
    architectures = [
        [], [32], [64], [32,32], [64,32], [64,64], [128,128]
    ]

    # Set hidden and out functions TODO: Add ability to control this
    hidden_functions = [nn.LeakyReLU, nn.Sigmoid]
    out_functions = [None, nn.Softmax]

    # Run experiments
    run_experiment(
        results_file = args.results_path,
        dataset_name = args.data,
        train_data = train_data,
        validation_data = validation_data,
        test_data = test_data,
        autoencoder_index = args.autoencoder_index,
        epochs = args.predictor_epochs,
        batch_size = args.predictor_batch_size,
        predictor_architectures = architectures,
        predictor_hidden_functions = hidden_functions,
        predictor_output_functions = out_functions,
        allowed_ae_parameters = {} #TODO: Add the ability to control this
    )

# When this file is executed independently, execute the main function
if __name__ == "__main__":
    main()