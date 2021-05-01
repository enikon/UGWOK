import tensorflow as tf
import os
import argparse
import numpy as np
import cv2


def files_to_numpy(x, dirs):
    res = [[cv2.imread(os.path.join(d, i + t)) for i in x] for d, t in dirs]
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default='../dataset')
    parser.add_argument('--labels', default='../labels')
    parser.add_argument('--val_split', default=0.15)
    parser.add_argument('--test_split', default=0.15)

    args = parser.parse_args()

    np.random.seed(42)

    _folder_dataset_names  = set([os.path.splitext(f)[0] for f in os.listdir(args.dataset) if f.endswith(('.jpg', '.png'))])
    _folder_dataset_labels = set([os.path.splitext(f)[0] for f in os.listdir(args.labels ) if f.endswith(('.jpg', '.png'))])

    _main_dataset_names = np.array(list(_folder_dataset_labels))
    _extra_dataset_names = np.array(list(_folder_dataset_names - _folder_dataset_labels))

    np.random.shuffle(_main_dataset_names)
    _main_length = len(_main_dataset_names)

    _test_length = int(args.test_split * _main_length)
    _val_length = int(args.val_split * _main_length)
    _train_length = _main_length - _test_length - _val_length

    _train_split_names = _main_dataset_names[:_train_length]
    _eval_split_names = _main_dataset_names[_train_length:_train_length+_val_length]
    _test_split_names = _main_dataset_names[_train_length+_val_length:]

    x_train, y_train = files_to_numpy(
        _train_split_names,
        [(args.dataset, '.jpg'), (args.labels, '.png')]
    )

    h=0

