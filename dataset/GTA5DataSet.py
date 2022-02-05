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

class GTA5DataSet(data.Dataset):
                 
    def __init__(self, root, list_path, info_json, crop_size=(321, 321), mean=(128, 128, 128), ignore_label=255, augmentation=None, max_iters=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.augmentation = augmentation
            
        self.img_paths = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        
        self.files = []

        self.mapping = np.array(info_json['label2train'], dtype=np.int)


        for name in self.img_paths:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
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

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label = label_mapping(label, self.mapping)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy() #, np.array(size), name 

