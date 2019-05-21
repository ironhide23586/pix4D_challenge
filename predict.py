# your implementation goes here
from glob import glob
import os
from PIL import Image

import torch
import numpy as np

from fcn_model.convnet import FCN
from fcn_model.preprocessing import preprocess_data_x


DATA_DIR = './images'
MODEL_SAVE_DIR = './trained_models'


class FCN_Infer:  # Class to encapsulate Segmentation inference over images of any size

    def __init__(self, resume_latest=True, model_native_input_side=256, model_native_output_side=129):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = FCN().to(self.device)
        existing_model_paths = glob(MODEL_SAVE_DIR + os.sep + '*')  # Resume from last checkpoint if available
        if len(existing_model_paths) > 0 and resume_latest:
            print('Found', len(existing_model_paths), ' pre-trained models! Resuming from the best one...')
            existing_ious = np.array([float(existing_model_paths[i].split('_')[-2][:-3])
                                      for i in range(len(existing_model_paths))])
            best_model_path = existing_model_paths[existing_ious.argmax()]
            print('Resuming from', best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
            print('Resumed! :D')
        else:
            print('No models found, randomly initializing weights...')
        self.model_native_input_side = model_native_input_side
        self.model_native_output_side = model_native_output_side

    def infer_img(self, im_rgb_hwc):  # High level method which wraps inference methods over images of any dims
        h, w, c = im_rgb_hwc.shape
        x_pp = preprocess_data_x(np.expand_dims(im_rgb_hwc, 0), normalize=False)
        num_w_patches = int(np.ceil(w / self.model_native_input_side))
        num_h_patches = int(np.ceil(h / self.model_native_input_side))
        patches = []
        for h_idx in range(num_h_patches):  # Splits image in to a grid of patches
            for w_idx in range(num_w_patches):
                start_h_idx = h_idx * self.model_native_input_side
                end_h_idx = start_h_idx + self.model_native_input_side
                if end_h_idx > h:
                    end_h_idx = h
                start_w_idx = w_idx * self.model_native_input_side
                end_w_idx = start_w_idx + self.model_native_input_side
                if end_w_idx > w:
                    end_w_idx = w
                patch = x_pp[:, :, start_h_idx:end_h_idx, start_w_idx:end_w_idx]
                patches.append(patch)

        with torch.no_grad():
            xs_torch = [torch.Tensor(p).to(self.device) for p in patches]
            preds_raw = [self.model.infer_softmax(x_torch) for x_torch in xs_torch]  # Inferencing on each patch
            preds_pp = [pred[0].cpu().numpy().argmax(axis=0) for pred in preds_raw]
            w_new = 0
            h_new = 0
            for w_idx in range(num_w_patches):
                w_new += preds_pp[w_idx].shape[1]

            for h_idx in range(num_h_patches):
                h_new += preds_pp[h_idx * num_w_patches].shape[0]

            pred = np.zeros([h_new, w_new]).astype(np.uint8)

            for h_idx in range(num_h_patches):  # Concatenating all the patches to get segmap of whole input image
                for w_idx in range(num_w_patches):
                    idx = w_idx + h_idx * num_w_patches
                    start_h_idx = h_idx * self.model_native_output_side
                    end_h_idx = start_h_idx + self.model_native_output_side
                    if end_h_idx > h_new:
                        end_h_idx = h_new
                    start_w_idx = w_idx * self.model_native_output_side
                    end_w_idx = start_w_idx + self.model_native_output_side
                    if end_w_idx > w_new:
                        end_w_idx = w_new
                    pred[start_h_idx:end_h_idx, start_w_idx:end_w_idx] = preds_pp[idx]
        pred[pred > 0] = 255

        # Nearest Neighbor upscaling to match input resolution
        mask = Image.fromarray(pred).resize((w, h), resample=Image.NEAREST)

        mask = 255 - np.array(mask)
        return mask


if __name__ == '__main__':
    im_pil = Image.open(DATA_DIR + os.sep + 'rgb.png')
    im = np.array(im_pil)

    fcn_inferer = FCN_Infer()
    im_segmask = fcn_inferer.infer_img(im)

    Image.fromarray(im_segmask).save(DATA_DIR + os.sep + 'pred.png')
