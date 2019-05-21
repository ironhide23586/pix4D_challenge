# your implementation goes here

import os
from glob import glob

import torch
import torch.nn as nn
import numpy as np

from fcn_model.convnet import FCN
from fcn_model.preprocessing import DataReader, preprocess_data


TOTAL_ITERS = 100000000
LEARN_RATE = 1e-4
BATCH_SIZE = 5

EVAL_FREQ = 50
EVAL_BATCH_SIZE = 100

DATA_DIR = './images'
MODEL_SAVE_DIR = './trained_models'
MODEL_NAME = 'pix4D_binary_segmentor_cnn'


def force_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_save_path = MODEL_SAVE_DIR + os.sep + MODEL_NAME

    data_reader_train = DataReader(DATA_DIR)
    data_reader_val = DataReader(DATA_DIR)
    x_val, y_val = data_reader_val.get_data(batch_size=EVAL_BATCH_SIZE)
    x_val_pp, y_val_truth = preprocess_data(x_val, y_val, normalize=False)

    x_val_torch = [torch.Tensor([x_val_pp[i]]).to(device) for i in range(EVAL_BATCH_SIZE)]

    model = FCN().to(device)

    loss_op = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    max_iou = -1
    force_makedir(MODEL_SAVE_DIR)

    existing_model_paths = glob(MODEL_SAVE_DIR + os.sep + '*')  # Resume from last checkpoint if available
    if len(existing_model_paths) > 0:
        print('Found', len(existing_model_paths), ' pre-trained models! Resuming from the best one...')
        existing_ious = np.array([float(existing_model_paths[i].split('_')[-2][:-3])
                                  for i in range(len(existing_model_paths))])
        best_model_path = existing_model_paths[existing_ious.argmax()]
        model.load_state_dict(torch.load(best_model_path))
        data_reader_train.total_batch_iters = int(best_model_path.split('iters')[0].split('_')[-1])
        print('Resuming from', best_model_path)

    with torch.no_grad():
        y_val_preds = np.array([model.infer_softmax(x_val_torch[i])[0].cpu().numpy().argmax(axis=0)
                                for i in range(EVAL_BATCH_SIZE)])
        y_val_preds = y_val_preds.astype(np.float32)
        intersection_count = (y_val_truth * y_val_preds).sum()
        union = y_val_truth + y_val_preds
        union[union > 0] = 1
        union_count = union.sum()
        iou = intersection_count / union_count
        print('IOU measure =', iou)
        if iou > max_iou:
            save_path = '_'.join([model_save_path, str(iou) + 'iou',
                                  str(data_reader_train.total_batch_iters) + 'iters']) + '.torchmodel'
            print('Saving model to', save_path)
            torch.save(model.state_dict(), save_path)
            max_iou = iou

    while data_reader_train.total_batch_iters < TOTAL_ITERS:
        x, y = data_reader_train.get_data(batch_size=BATCH_SIZE)
        x_pp, y_pp = preprocess_data(x, y, normalize=False)
        x_torch = torch.Tensor(x_pp).to(device)
        y_torch = torch.LongTensor(y_pp).to(device)

        outputs = model(x_torch)
        loss = loss_op(outputs, y_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Iter', data_reader_train.total_batch_iters, 'loss =', loss)

        if data_reader_train.total_batch_iters % EVAL_FREQ == 0:
            with torch.no_grad():
                y_val_preds = np.array([model.infer_softmax(x_val_torch[i])[0].cpu().numpy().argmax(axis=0)
                                        for i in range(EVAL_BATCH_SIZE)])
                y_val_preds = y_val_preds.astype(np.float32)
                intersection_count = (y_val_truth * y_val_preds).sum()
                union = y_val_truth + y_val_preds
                union[union > 0] = 1
                union_count = union.sum()
                iou = intersection_count / union_count
                print('IOU measure =', iou)
                if iou > max_iou:
                    save_path = '_'.join([model_save_path, str(iou) + 'iou',
                                          str(data_reader_train.total_batch_iters) + 'iters']) + '.torchmodel'
                    print('Saving model to', save_path)
                    torch.save(model.state_dict(), save_path)
                    max_iou = iou