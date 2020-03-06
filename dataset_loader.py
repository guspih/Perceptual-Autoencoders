import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pickle
import numpy as np
import scipy.io as sio

class PreprocessDataset(Dataset):
    '''
    A Dataset that must be fractioned and each fraction need to be preprocessed
    Args:
        datas ([[any]]): A list of the data where each data contains datapoints
        preprocess (f(any)->tensor): Function from datapointsto tensor
    '''
    def __init__(self, datas, preprocess):
        self.datas = datas
        self.preprocess = preprocess

    def __getitem__(self, index):
        return tuple(self.preprocess(data[index]) for data in self.datas)

    def __len__(self):
        return len(self.datas[0])


def split_data(datas, split_sizes=[0.8, 0.2]):
    '''
    Splits the dataset into sets of the given proportions
    Args:
        datas ([tensor]): The data to be split
        split_sizes ([int]): The relative sizes of the splits
    Returns ([[tensor]]): The list of splits
    '''
    start_index = 0
    splits = []
    split_sizes = [split_size/sum(split_sizes) for split_size in split_sizes]
    for split_size in split_sizes:
        end_index = min(
            int(datas[0].size(0)*split_size)+start_index, datas[0].size(0)
        )
        splits.append(
            [data[start_index:end_index] for data in datas]
        )
        start_index = end_index
    return splits

def load_pickled_gym_data(path_to_data, val_split=0.2):
    '''
    Takes pickled gym data and prepares it for pytorch use
    Args:
        path_to_data (str): Path to the .pickle file with data
        val_split (float): What fraction of data to use for validation
    Returns ({data}): A dict with data in training and validation splits
    '''
    assert val_split <= 1 and val_split >= 0, \
        'val_split must be between 0 and 1'
    
    data = pickle.load(open(path_to_data, 'rb'))
    parameters = data['parameters']
    data_size = parameters['rollouts']*parameters['timesteps_per_rollout']
    val_index = data_size - int(data_size*val_split)
    val_index = val_index - (val_index % parameters['timesteps_per_rollout'])

    for key, value in data.items():
        if key == 'parameters':
            continue
        assert len(value) == data_size, \
            'non-parameter data should contain data_size ({}) entries'.format(
                data_size
            )
        if key == 'imgs':
            value = np.transpose(value, (0,3,1,2))
        if (np.array(value).dtype.kind in ['f','u','i']):
            value = torch.from_numpy(np.array(value, dtype=np.float32))
        train, valid = value[:val_index], value[val_index:data_size]
        data[key] = train, valid
    return data

def load_lunarlander_data(path_to_data, keep_off_screen=True):
    '''
    Takes pickled gym LunarLander-v2 data and prepares it for pytorch use
    Args:
        path_to_data (str): Path to the .pickle file with data
        keep_off_screen (bool): Whether to keep images with lander off-screen
    Returns (tensor, tensor): The images and corresponing lander positions
    '''
    
    data = load_pickled_gym_data(path_to_data, 0)
    images = data['imgs'][0].float()
    labels = data['observations'][0]
    labels = labels.narrow(1,0,2).float()
    if not keep_off_screen:
        #Remove data where the lander is off screen (-1<=x<=1 & -0.5<=y<=1.5)
        condition = (
            (labels[:,0]<=1) & (labels[:,0]>=-1) &
            (labels[:,1]<=1.5) & (labels[:,1]>=-0.5)
        )
        labels = labels[condition, :]
        images = images[condition, :]
    return images, labels

def load_svhn_data(path_to_data):
    '''
    Reads and returns the data for the svhn dataset
    Args:
        path_to_data (str): Path to the binary file containing images and labels
    Returns (tensor, tensor): The images wrap-padded to be 64x64 and the labels
    '''

    data = sio.loadmat(path_to_data)
    images = data['X']
    images = np.transpose(images, (3,2,0,1))
    images = np.pad(images, ((0,0),(0,0),(0,32),(0,32)), mode='wrap')
    images = images/255
    images = torch.from_numpy(images).float()
    labels = data['y']
    labels = labels.reshape((-1))
    labels = labels-1
    labels = np.eye(10)[labels]
    labels = torch.from_numpy(labels).float()
    return images, labels

def load_stl_data(path_to_images, path_to_labels=None):
    '''
    Reads and returns the images and labels for the STL-10 dataset
    Args:
        path_to_images (str): Path to the binary file containing images
        path_to_labels (str): Path to the binary file containing labels 
    Returns (tensor, tensor): The images with channels first and labels
    '''

    with open(path_to_images, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = images/255
        images = torch.from_numpy(images).float()

    if not path_to_labels is None:
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            labels = labels-1
            labels = np.eye(10)[labels]
            labels = torch.from_numpy(labels).float()
    else:
        labels = None
    
    return images, labels