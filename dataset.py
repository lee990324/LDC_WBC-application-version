import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json

MODEL_DIR = '16/16_model.pth'

DATASET_NAMES = [
    'TRAIN',
    'VAL',
    'TEST'
]  # 8

def dataset_info(dataset_name):
    config = {
        'TRAIN': {
            'img_height': 720, #720 # 1088
            'img_width': 1280, # 1280 5 1920
            'data_dir': 'C:/Users/gram15/Desktop/Working_dir/LDC_WBC-application-version/data',
            'yita': 0.5
        },
        'VAL': {
            'img_height': 720,#
            'img_width': 1280,# 512
            'data_dir': 'C:/Users/gram15/Desktop/Working_dir/LDC_WBC-application-version/data',
            'yita': 0.5
        },
        'TEST': {
            'img_height': 512,#
            'img_width': 512,# 512
            'data_dir': 'data',  # mean_rgb
            'yita': 0.5
        }
    }
    return config[dataset_name]



class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 arg=None
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.args = arg
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        if self.test_data == "TEST":
            # Test
            images_path = os.listdir(data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # Val
            images_path = os.path.join(data_root,
                                        'imgs/val')
            labels_path = os.path.join(data_root,
                                        'edge_maps/val')

            for file_name_ext in os.listdir(images_path):
                file_name = os.path.splitext(file_name_ext)[0]
                sample_indices.append(
                    (os.path.join(images_path, file_name + '.jpg'),
                        os.path.join(labels_path, file_name + '.png'),))
            
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper() == 'TEST' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx] if len(self.data_index[0]) > 1 else self.data_index[0][idx - 1]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "TEST" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # base dir
        if self.test_data.upper() == 'TRAIN':
            img_dir = os.path.join(self.data_root, 'imgs', 'test')
            gt_dir = os.path.join(self.data_root, 'edge_maps', 'test')
        elif self.test_data.upper() == 'TEST':
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        if not self.test_data == "TEST":
            label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "TEST":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width, img_height))
            gt = None

        # Make images and labels at least 512 by 512
        if img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height))  # 512
            gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height))  # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
            img_width = ((img.shape[1] // 8) + 1) * 8
            img_height = ((img.shape[0] // 8) + 1) * 8
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            pass
        # # For FPS
        # img = cv2.resize(img, (496,320))
        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class TrainDataset(Dataset):
    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg

        self.data_index = self._build_index()

    def _build_index(self):

        data_root = os.path.abspath(self.data_root)
        sample_indices = []

        images_path = os.path.join(data_root,
                                    'imgs/train')
        labels_path = os.path.join(data_root,
                                    'edge_maps/train')

        for directory_name in os.listdir(images_path):
            image_directories = os.path.join(images_path, directory_name)
            for file_name_ext in os.listdir(image_directories):
                file_name = os.path.splitext(file_name_ext)[0]
                sample_indices.append(
                    (os.path.join(images_path, directory_name, file_name + '.jpg'),
                        os.path.join(labels_path, directory_name, file_name + '.png'),)
                )

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.  # for LDC input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w, _ = img.shape
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None  # 448# MDBD=480 BIPED=480/400 BSDS=352

        # for BSDS 352/BRIND
        if i_w > crop_size and i_h > crop_size:  # later 400, before crop_size
            i = random.randint(0, i_h - crop_size)
            j = random.randint(0, i_w - crop_size)
            img = img[i:i + crop_size, j:j + crop_size]
            gt = gt[i:i + crop_size, j:j + crop_size]

        # # for BIPED/MDBD
        # if i_w> 420 and i_h>420: #before np.random.random() > 0.4
        #     h,w = gt.shape
        #     if np.random.random() > 0.4: #before i_w> 500 and i_h>500:
        #
        #         LR_img_size = crop_size #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
        #         i = random.randint(0, h - LR_img_size)
        #         j = random.randint(0, w - LR_img_size)
        #         # if img.
        #         img = img[i:i + LR_img_size , j:j + LR_img_size ]
        #         gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
        #     else:
        #         LR_img_size = 300#208  # l BIPED=208-352, # MDBD= 352-480- BSDS= 176-320
        #         i = random.randint(0, h - LR_img_size)
        #         j = random.randint(0, w - LR_img_size)
        #         # if img.
        #         img = img[i:i + LR_img_size, j:j + LR_img_size]
        #         gt = gt[i:i + LR_img_size, j:j + LR_img_size]
        #         img = cv2.resize(img, dsize=(crop_size, crop_size), )
        #         gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        else:
            # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # # BRIND
        # gt[gt > 0.1] +=0.2#0.4
        # gt = np.clip(gt, 0., 1.)
        # for BIPED
        gt[gt > 0.2] += 0.6  # 0.5 for BIPED
        gt = np.clip(gt, 0., 1.)  # BIPED
        # # for MDBD
        # gt[gt > 0.3] +=0.7#0.4
        # gt = np.clip(gt, 0., 1.)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt