import pickle
import random
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


def adc(image):
    """Normalize dwi by image with b_value == 0"""
    max_val = 65535
    image = torch.clip(image, 1, max_val)
    image = torch.log(image)
    image = image[:] - image[0:1]
    return image[1:]


def remove_ncf(img):
    """
    Remove NCF.
    """
    NCF = 210
    tmp_img = img**2 - NCF
    tmp_img[tmp_img < 0] = 0
    return np.sqrt(tmp_img)


def deform_class(label, oldlabel2newlabel):
    """
    Map no label class to benign.
    Here we assume no label dwi is, probabilistically, more similar to benign than malignant.
    """
    label = oldlabel2newlabel[label]
    if label == 0:
        return label
    else:
        return label - 1


def fix_shape(image, max_h, max_w):
    """Resize all inputs to constant shape"""
    canvas = np.zeros((max_h, max_w, image.shape[-1]))
    h, w, _ = image.shape
    canvas[
        (max_h - h) // 2 : (max_h - h) // 2 + h, (max_w - w) // 2 : (max_w - w) // 2 + w
    ] = image
    return canvas


class DWIDataset(Dataset):
    def __init__(self, data, args, transform=None):
        self.transform = transform
        self.data_num = len(data)
        self.data = data
        self.args = args
        self.size = 96
        self.mean = None
        self.std = None

    def __len__(self):
        return self.data_num

    def preprocess(self, image, label):
        label = deform_class(label, {"N": 0, "B": 1, "M": 2})
        image = image.transpose(1, 2, 0)
        image = fix_shape(image, self.size, self.size)
        if self.args.remove_ncf:
            image = remove_ncf(image)
        return image, label

    def __getitem__(self, idx):
        pass

    def set_stats_from_data(self):
        if self.args.adc:
            channels = 4
        else:
            channels = 5
        img_sum_each_b = np.zeros(channels)
        img_square_sum_each_b = np.zeros(channels)
        pixel_count = 0
        for out_paths in self.data:
            for out_path in out_paths:
                f = open(out_path, "rb")
                img, tumor_class = pickle.load(f)
                img, _ = self.preprocess(img, tumor_class)
                assert img.shape[-1] == 5
                if self.args.adc:
                    img = adc(torch.tensor(img.transpose(2, 0, 1))).numpy().transpose(1, 2, 0)
                img_sum_each_b += np.sum(np.sum(img, axis=0), axis=0)
                img_square_sum_each_b += np.sum(np.sum(img**2, axis=0), axis=0)
                pixel_count += np.count_nonzero(img[:, :, 0])
        mean = img_sum_each_b / pixel_count
        std = np.sqrt(img_square_sum_each_b / pixel_count - mean**2)
        self.mean = mean
        self.std = std
        print(mean, std)
        return mean, std

    def set_stats(self, mean, std):
        self.mean = np.float32(mean)
        self.std = np.float32(std)


class DWIDataset3D5Slices(DWIDataset):
    def __init__(self, data, args, transform=None, mode="train"):
        super().__init__(data, args, transform=transform)
        self.mode = mode

    def __getitem__(self, idx):
        out_paths = self.data[idx]
        imgs = []
        classes = []
        for out_path in out_paths:
            f = open(out_path, "rb")
            img, tumor_class = pickle.load(f)
            img, label = self.preprocess(img, tumor_class)
            imgs.append(img)
            classes.append(label)
        label = classes[2]
        imgs = np.array(imgs)
        imgs = imgs.transpose(3, 0, 1, 2)
        c, d, h, w = imgs.shape
        assert imgs.shape[0] == 5
        randindex = random.randint(0, d - 5)
        if self.mode == "train":
            imgs = imgs[:, randindex : randindex + 5]
        elif self.mode == "test":
            imgs = imgs[:, d // 2 - 2 : d // 2 + 3]
        out_data = torch.tensor(np.float32(imgs))
        if self.transform:
            out_data = self.transform(out_data)
        if self.args.adc:
            out_data = adc(out_data)
        assert self.mean is not None
        assert self.std is not None
        if self.mean is not None and self.std is not None:
            out_data = (out_data - self.mean[:, None, None, None]) / self.std[
                :, None, None, None
            ]
        return out_data, label, out_paths

    def set_stats_from_data(self):
        if self.args.adc:
            channels = 4
        else:
            channels = 5
        img_sum_each_b = np.zeros(channels)
        img_square_sum_each_b = np.zeros(channels)
        pixel_count = 0
        for out_paths in self.data:
            for out_path in out_paths:
                f = open(out_path, "rb")
                img, tumor_class = pickle.load(f)
                img, _ = self.preprocess(img, tumor_class)
                assert img.shape[-1] == 5
                if self.args.adc:
                    img = adc(torch.tensor(img.transpose(2, 0, 1))).numpy().transpose(1, 2, 0)
                img_sum_each_b += np.sum(np.sum(img, axis=0), axis=0)
                img_square_sum_each_b += np.sum(np.sum(img**2, axis=0), axis=0)
                pixel_count += np.count_nonzero(img[:, :, 0])
        mean = img_sum_each_b / pixel_count
        std = np.sqrt(img_square_sum_each_b / pixel_count - mean**2)
        self.mean = mean
        self.std = std
        return mean, std


class DWIDataset3D(DWIDataset):
    def __init__(self, data, args, transform=None):
        super().__init__(data, args, transform=transform)

    def __getitem__(self, idx):
        out_paths = self.data[idx]
        imgs = []
        classes = []
        for out_path in out_paths:
            f = open(out_path, "rb")
            img, tumor_class = pickle.load(f)
            img, label = self.preprocess(img, tumor_class)
            imgs.append(img)
            classes.append(label)
        assert len(set(classes)) == 1
        label = [classes[0]]
        imgs = np.array(imgs)
        imgs = imgs.transpose(3, 0, 1, 2)
        assert imgs.shape[0] == 5
        out_data = np.float32(imgs)
        out_data = torch.tensor(out_data)
        if self.transform:
            out_data = self.transform(out_data)
        if self.args.adc:
            out_data = adc(out_data)
        assert self.mean is not None
        assert self.std is not None
        if self.mean is not None and self.std is not None:
            out_data = (out_data - self.mean[:, None, None, None]) / self.std[
                :, None, None, None
            ]
        return out_data, np.array(label), out_paths
