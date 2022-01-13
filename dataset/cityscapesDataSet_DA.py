import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # in the list_path file of paths format [ name_of_folder/name_of_image ] -> img_ids list of paths format [name_of_image]
        self.img_ids = [i_id.strip().split("/")[1] for i_id in open(list_path)] 
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root , "images", name)
            self.files.append({
                "img": img_file,
                "name": name
            })




    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        return len(self.files)


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        datafiles = self.files[index]
        
        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BILINEAR)
        
        # convert as array
        image = np.asarray(image, np.float32)

        # change to BGR
        image = image[:, :, ::-1]  
        # normalise
        image -= self.mean
        # transpose the image from HWC-layout (height, width, channels) -> (CHW layout)
        image = image.transpose((2, 0, 1)) # see: https://github.com/isl-org/MiDaS/issues/79 
        size = image.shape
        
        return image.copy() #, np.array(size), name
