# your implementation goes here

import os
from PIL import Image

import torch
import torch.nn as nn
import numpy as np


TOTAL_ITERS = 100000000
LEARN_RATE = 1e-4
BATCH_SIZE = 2

DATA_DIR = './images'


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ))
        self.layers.append(nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))
        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=1),
            nn.Conv2d(8, 2, kernel_size=5, stride=1, padding=2),
            nn.Softmax2d()
        ))

        self.all_layers = nn.Sequential(
            self.layers[0],
            self.layers[1],
            self.layers[2],
            self.layers[3]
        )

    def forward(self, x):
        out = self.all_layers(x)
        return out


class DataReader:

    def __init__(self, dir):
        self.x_path = dir + os.sep + 'rgb.png'
        self.y_path = dir + os.sep + 'gt.png'
        self.x_im_pil = Image.open(self.x_path)
        self.y_im_pil = Image.open(self.y_path)
        self.total_batch_iters = 0

    def get_data(self, batch_size=1, patch_w=256, patch_h=256, label_w=129, label_h=129):
        x = []
        y = []
        w_max = self.x_im_pil.width - patch_w
        h_max = self.y_im_pil.height - patch_h
        while len(x) < batch_size:
            w_start = np.random.randint(w_max)
            h_start = np.random.randint(h_max)
            random_crop_bbox = (w_start, h_start, w_start + patch_w, h_start + patch_h)
            random_rotate_angle = np.random.randint(360)
            x_elem = self.x_im_pil.crop(box=random_crop_bbox).rotate(random_rotate_angle)
            y_elem = self.y_im_pil.crop(box=random_crop_bbox).rotate(random_rotate_angle).resize((label_w, label_h),
                                                                                                 Image.NEAREST)

            y_tmp = np.array(y_elem)
            rotation_mask = np.ones([y_elem.height, y_elem.height, 3]).astype(np.uint8)
            rotation_mask[rotation_mask > 0] = 255
            rotation_mask_pil = Image.fromarray(rotation_mask).rotate(random_rotate_angle)
            rotation_mask = 255 - np.array(rotation_mask_pil)
            y_tmp = y_tmp.astype(np.float32) + rotation_mask
            y_tmp[y_tmp > 0] = 255
            y_tmp = y_tmp.astype(np.uint8)

            mask = y_tmp[:, :, 0]
            postive_frac = mask[mask < 255].shape[0] / mask[mask==255].shape[0]
            if postive_frac > 0.:
                x.append(np.array(x_elem)[:, :, :3])
                y.append(y_tmp)
                self.total_batch_iters += 1
        x = np.array(x)
        y = np.array(y)
        return x, y


def mono2multi_channel_tiler(v, channels=3):
    v_tiled = np.array([np.rollaxis(np.tile(v_, [channels, 1, 1]), 0, 3) for v_ in v])
    return v_tiled


def preprocess_data(x, y, normalize=True):
    x_pp = x / 255.
    if normalize:
        # channel_means = np.array([x_pp_.mean(axis=0).mean(axis=0) for x_pp_ in x_pp])
        pixelwise_means = np.array([x_pp_.mean(axis=2) for x_pp_ in x_pp])
        pixelwise_means_tiled = mono2multi_channel_tiler(pixelwise_means)
        pixelwise_stds = np.array([x_pp_.std(axis=2) for x_pp_ in x_pp])
        pixelwise_stds_tiled = mono2multi_channel_tiler(pixelwise_stds)
        filt = pixelwise_stds_tiled > 0
        x_pp[filt] = (x_pp[filt] - pixelwise_means_tiled[filt]) / pixelwise_stds_tiled[filt]
    masks = np.array([np.rollaxis(np.array([y_[:, :, 0], 255 - y_[:, :, 0]]), 0, 3) for y_ in y])
    masks[masks > 0] = 1  # channel 1 is one on building pixels
    masks = masks.astype(np.float32)
    x_pp = np.rollaxis(x_pp, 3, 1)  # changing from NHWC to NCHW for Torch
    masks = np.rollaxis(masks, 3, 1)
    return x_pp, masks


if __name__ == '__main__':

    data_reader = DataReader(DATA_DIR)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FCN().to(device)

    loss_op = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    while data_reader.total_batch_iters < TOTAL_ITERS:
        x, y = data_reader.get_data(batch_size=BATCH_SIZE)
        x_pp, y_pp = preprocess_data(x, y, normalize=False)
        x_torch = torch.Tensor(x_pp).to(device)
        y_torch = torch.Tensor(y_pp).to(device)

        outputs = model(x_torch)
        loss = loss_op(outputs, y_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        k =0



