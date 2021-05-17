import os
import numpy as np
import argparse
import tensorflow as tf

from unet_main import evaluate_unet, predict_unet, train_unet
from datasets import extract_x_y_mask, save_split, save_masks_cmp

# FOR DEBUGGING
# tf.config.experimental_run_functions_eagerly(True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default='../dataset')
    parser.add_argument('--labels', default='../labels')
    parser.add_argument('--load_split', default=False, action='store_true')
    parser.add_argument('--split_dir', default='../split') #filenames
    parser.add_argument('--val_split', default=0.15)
    parser.add_argument('--test_split', default=0.15)
    args = parser.parse_args()

    _folder_dataset_names = set(
        [os.path.splitext(f)[0] for f in os.listdir(args.dataset) if f.endswith(('.jpg', '.png'))])
    _folder_dataset_labels = set(
        [os.path.splitext(f)[0] for f in os.listdir(args.labels) if f.endswith(('.jpg', '.png'))])

    _main_dataset_names = np.sort(list(_folder_dataset_labels))
    _extra_dataset_names = np.sort(list(_folder_dataset_names - _folder_dataset_labels))

    # np.random.seed(42)
    np.random.shuffle(_main_dataset_names)
    _main_length = len(_main_dataset_names)

    _test_length = int(args.test_split * _main_length)
    _val_length = int(args.val_split * _main_length)
    _train_length = _main_length - _test_length - _val_length

    _train_split_names = _main_dataset_names[:_train_length]
    _val_split_names = _main_dataset_names[_train_length:_train_length + _val_length]
    _test_split_names = _main_dataset_names[_train_length + _val_length:]

    # Saving splits
    save_split(_train_split_names, os.path.join(args.split_dir, 'train'))
    save_split(_val_split_names, os.path.join(args.split_dir, 'val'))
    save_split(_test_split_names, os.path.join(args.split_dir, 'test'))

    sets = [
        extract_x_y_mask(_train_split_names, args, True),
        extract_x_y_mask(_val_split_names, args),
        extract_x_y_mask(_test_split_names, args)
    ]

    model, model_history = train_unet(sets)
    evaluate_unet(model, sets)
    pred_masks = predict_unet(model, sets[2][0])
    save_masks_cmp(sets[2][0], pred_masks, sets[2][1])

    h = 0
