import os
from PIL import Image

import numpy as np


class DataReader:  # This class encapsulates generation of crops for training

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
            if postive_frac > 0.:  # Rejecting images with no building mask
                x.append(np.array(x_elem)[:, :, :3])
                y.append(y_tmp)
                self.total_batch_iters += 1
        x = np.array(x)
        y = np.array(y)
        return x, y  # Random crops of input images + labels also rotated randomly


def mono2multi_channel_tiler(v, channels=3):  # Tiles a single channel image to multi-channel
    v_tiled = np.array([np.rollaxis(np.tile(v_, [channels, 1, 1]), 0, 3) for v_ in v])
    return v_tiled


def preprocess_data(x, y, normalize=False):
    x_pp = preprocess_data_x(x, normalize=normalize)
    masks = np.array([np.rollaxis(np.array([y_[:, :, 0], 255 - y_[:, :, 0]]), 0, 3) for y_ in y])
    masks[masks > 0] = 1  # channel 1 is one on building pixels
    masks = masks.astype(np.float32)
    masks = np.rollaxis(masks, 3, 1)[:, 1, :, :]
    return x_pp, masks


def preprocess_data_x(x, normalize=False):
    x_pp = x / 255.
    x_pp = x_pp[:, :, :, :3]
    if normalize:
        # channel_means = np.array([x_pp_.mean(axis=0).mean(axis=0) for x_pp_ in x_pp])
        pixelwise_means = np.array([x_pp_.mean(axis=2) for x_pp_ in x_pp])
        pixelwise_means_tiled = mono2multi_channel_tiler(pixelwise_means)
        pixelwise_stds = np.array([x_pp_.std(axis=2) for x_pp_ in x_pp])
        pixelwise_stds_tiled = mono2multi_channel_tiler(pixelwise_stds)
        filt = pixelwise_stds_tiled > 0
        x_pp[filt] = (x_pp[filt] - pixelwise_means_tiled[filt]) / pixelwise_stds_tiled[filt]
    x_pp = np.rollaxis(x_pp, 3, 1)  # changing from NHWC to NCHW for Torch
    return x_pp