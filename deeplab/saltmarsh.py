from torch.utils import data
import os
import numpy as np
import scipy.misc as m
from PIL import Image
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
#this class is based on Cityscapes Dataset in the given repo..
#this class is based on Cityscapes Dataset in the given repo..

class SaltmarshSegmentation(data.Dataset):
    NUM_CLASSES = 9

    def __init__(self, args,root=r"Data", split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.images = []
        self.masks=[]

        self.set_data_names()

        #self.void_classes = [0]
        self.valid_classes = [0,1,2,3,4,5,6,7,8]
        self.class_names = ['Background','Limonium', 'Spartina', 'Batis', 'Other', 'Spart_dead', \
                            'Juncus', 'Sacricornia', 'Borrichia']

        #self.ignore_index = 255
        #self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        img_path = self.root+"/"+self.split+"/"+self.images[index]
        lbl_path = self.root+"/"+self.split+"/"+self.masks[index]

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)
    def set_data_names(self):  
        for file in os.listdir(self.root+"/"+self.split+"/"):
            if (file.endswith("mask.png")):
                    self.masks.append(file)
                    s=file.split('_')
                    imgname="_".join(s[:-1])+".jpg"
                    self.images.append(imgname)
        return True

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)