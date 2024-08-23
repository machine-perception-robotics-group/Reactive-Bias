# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import random
import cv2
import torch
import numpy as np


import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torchvision.transforms.functional as F

class MyRandomSizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), i, j, h, w, self.size, self.interpolation

class MyRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), True
        return img, False

class MyImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None,hname=None):
        super(MyImageFolder, self).__init__(root, transform=transform)
        self.hname =hname
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # 入力画像と教師ラベルの読み込み
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Attention mapの読み込み
        at_path = path.split("/")
        # print(at_path[6])
        at_path[6] = self.hname #人の知見
        #print(at_path)
        # at_path[8],_ = at_path[8].split(".")
        # del at_path[6]
        at_path = "/".join(at_path)
        # at_path = at_path+".png"
        # print(at_path)
        # print(os.path.isfile(at_path)
        at_map = self.loader(at_path) if os.path.isfile(at_path) else None
        # at_map = None
        # print(at_map)
        # data augumentation
        # if self.transform is not None:
        # この中で Random Crop + Flip を実装する
        sample, i, j, h, w, size, interpolation = MyRandomSizedCrop(224)(sample)
        sample, flag = MyRandomHorizontalFlip()(sample)
        if at_map != None:
            at_map = F.resized_crop(at_map, i, j, h, w, (14, 14), interpolation)
            if flag == True:
                at_map = F.hflip(at_map)
            at_map = transforms.Grayscale()(at_map)
            at_map = transforms.ToTensor()(at_map)
            at_map = torch.from_numpy(cv2.filter2D(at_map.numpy(), -1, np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])))
            is_atmap = 1
        else:
            at_map = torch.ones(1, 14, 14)*np.NaN
            is_atmap = 0
            print("NAN")
        # sample = self.transform(sample)
        sample = transforms.ToTensor()(sample)
        sample = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(sample)
        return sample, at_map, target, is_atmap

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'CUB':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif args.data_set == 'CUBATT':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if is_train:
            dataset = MyImageFolder(root, transform=transform,hname=args.hname)
        else:
            dataset = datasets.ImageFolder(root,transform=transform)
        nb_classes = 200

    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
