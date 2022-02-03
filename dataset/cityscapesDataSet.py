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

def label_mapping(input, mapping):
   '''
   Given the input version of the labels (array of elements) performs a mapping using a mapping function and outputs the mapped labels
   input = array of labels
   mapping = array with format [oldlabel , newlabel]
   output = array of mapped labels
   '''
   output = np.copy(input)
   for ind in range(len(mapping)):
     output[input == mapping[ind][0]] = mapping[ind][1]
   return np.array(output, dtype=np.int64)

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, info_json, crop_size=(321, 321), mean=(128, 128, 128), ignore_label=255, augmentation=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.augmentation = augmentation

        # in the list_path file of paths format [ name_of_folder/name_of_image ] -> img_ids list of paths format [name_of_image]
        self.img_ids = [i_id.strip().split("/")[1] for i_id in open(list_path)] 
        
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root , "images", name)
            label_file = osp.join(self.root , "labels", name).replace("leftImg8bit", "gtFine_labelIds")
            self.files.append({
                "img": img_file,
                "label":label_file,
                "name": name
            })

        self.mapping = np.array(info_json['label2train'], dtype=np.int)



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
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
         
        if not self.augmentation == None:
            if np.random.rand() < self.augmentation["prob"]:
               # horizontal flipping
               hor_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
               image = hor_flip(image)
               label = hor_flip(label)
               # gaussian blur
               blurred = torchvision.transforms.GaussianBlur(kernel_size = self.augmentation["kernel_size"], sigma = self.augmentation["sigma"])
               image = blurred(image)

        # resize
        image = image.resize(self.crop_size, Image.BILINEAR)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        # convert as array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        
        # map the labels
        label = label_mapping(label, self.mapping)

        # change to BGR
        image = image[:, :, ::-1]  
        # normalise
        image -= self.mean
        # transpose the image from HWC-layout (height, width, channels) -> (CHW layout)
        image = image.transpose((2, 0, 1)) # see: https://github.com/isl-org/MiDaS/issues/79 

        return image.copy(), label.copy()
