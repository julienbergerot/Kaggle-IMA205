from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


# Dataset for training and validation
class MelanomaImageDataset(Dataset):
    def __init__(self, paths, labels, folder : str, transform):
        """
            paths : list of paths to the images
            labels : list of the labels (between 1 and 8)
            folder : folder containing the data
            transform : the transformation to apply
        """
        self.paths = paths
        self.labels = labels  
        self.transform = transform
        self.folder = folder
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # choose the file
        file = self.paths[idx]
        label = self.labels[idx]-1 # as the label provided are between 1 and 8 (included)
        # open the file
        inp = Image.open(os.path.join(self.folder,file+".jpg"))
        # Apply transformation
        if self.transform :
            inp = self.transform(inp)
                
        return inp, label


# Dataset for testing and kaggle submission
class MelanomaImageDatasetTest(Dataset):
    def __init__(self, paths, folder : str, transform):
        """
            paths : list of paths to the images
            folder : folder containing the data
            transform : the transformation to apply
        """
        self.paths = paths
        self.transform = transform
        self.folder = folder
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # choose the file
        file = self.paths[idx]
        # open the file
        inp = Image.open(os.path.join(self.folder,file+".jpg"))
        # open the file
        inp = Image.open(os.path.join(self.folder,file+".jpg"))
        # Apply transformation
        if self.transform :
            inp = self.transform(inp)
        # returns the image and its ID
        return inp, self.paths[idx]