import tensorflow as tf
import os
import argparse
import numpy as np
import cv2

from unet_main import unet_main


def files_to_numpy(x, dirs):
    res = [np.array([
        cv2.resize(
            cv2.imread(os.path.join(d, i + t)),
            (128, 128)
        )
        for i in x]) for d, t in dirs]
    return res


def labelise(x):
    p = np.array([[[0, 0, -128]]])
    r = np.array([1.0, 1.0, 1.0])

    label = np.array([1-np.abs(np.sign(np.dot(i+p, r))) for i in x])
    mask = np.array([np.abs(np.sign(np.dot(i, r))) for i in x])

    return label, mask


def extract_x_y_mask(x, args):
    x_train, y_train = files_to_numpy(
        x,
        [(args.dataset, '.jpg'), (args.labels, '.png')]
    )
    y_train, mask_train = labelise(y_train)

    return x_train, y_train, mask_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default='../dataset')
    parser.add_argument('--labels', default='../labels')
    parser.add_argument('--val_split', default=0.15)
    parser.add_argument('--test_split', default=0.15)

    args = parser.parse_args()

    _folder_dataset_names  = set([os.path.splitext(f)[0] for f in os.listdir(args.dataset) if f.endswith(('.jpg', '.png'))])
    _folder_dataset_labels = set([os.path.splitext(f)[0] for f in os.listdir(args.labels ) if f.endswith(('.jpg', '.png'))])

    _main_dataset_names = np.sort(list(_folder_dataset_labels))
    _extra_dataset_names = np.sort(list(_folder_dataset_names - _folder_dataset_labels))

    np.random.seed(42)
    np.random.shuffle(_main_dataset_names)
    _main_length = len(_main_dataset_names)

    _test_length = int(args.test_split * _main_length)
    _val_length = int(args.val_split * _main_length)
    _train_length = _main_length - _test_length - _val_length

    _train_split_names = _main_dataset_names[:_train_length]
    _val_split_names = _main_dataset_names[_train_length:_train_length+_val_length]
    _test_split_names = _main_dataset_names[_train_length+_val_length:]

    #x_train, y_train, mask_train = extract_x_y_mask(_train_split_names, args)
    #x_val, y_val, mask_val = extract_x_y_mask(_train_split_names, args)

    sets = [
        extract_x_y_mask(_train_split_names, args),
        extract_x_y_mask(_val_split_names, args),
        extract_x_y_mask(_test_split_names, args)
    ]

    unet_main(sets)

    h=0

