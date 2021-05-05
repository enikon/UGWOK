import os
import tensorflow as tf
import argparse
import numpy as np
import cv2
import albumentations as albu
from unet_main import unet_main


IMAGE_DIM_SIZE = 128
AUGMENTATION_MULTIPLIER = 9


transform_train = albu.Compose([
    albu.Flip(),
    albu.RandomRotate90(),
    albu.RandomResizedCrop(height=IMAGE_DIM_SIZE, width=IMAGE_DIM_SIZE, scale=(0.5, 1.0), p=0.5)
])


def read_images(imges_names, path, dim_size):
    return np.array([cv2.resize(
        cv2.imread(os.path.join(path[0], i + path[1])),
        dim_size
    ) for i in imges_names])


def augment_data(images, masks, times=1):
    resultImages = []
    resultMasks = []

    for i in range(times):
        for img, mask in zip(images, masks):
            augmentation = {"image": img, "mask": mask}
            augmentation = transform_train(**augmentation)
            resultImages.append(augmentation["image"])
            resultMasks.append(augmentation["mask"])

    return np.array(resultImages), np.array(resultMasks)


def binarize_masks(masks):
    # 255 -> red, 0 -> other colors
    red = (0, 0, 128)
    return np.array([cv2.inRange(m, red, red) for m in masks])


def files_to_numpy(x, images_path, masks_path, is_train_set):
    resizeDims = (IMAGE_DIM_SIZE, IMAGE_DIM_SIZE)

    realImages = read_images(x, images_path, resizeDims)
    realMasks = read_images(x, masks_path, resizeDims)

    if is_train_set:
        augmentedImages, augmentedMasks = augment_data(realImages, realMasks, AUGMENTATION_MULTIPLIER)
        resultImages = np.concatenate((realImages, augmentedImages))
        resultMasks = np.concatenate((realMasks, augmentedMasks))
    else:
        resultImages = realMasks
        resultMasks = realMasks

    return [resultImages, resultMasks]


def labelise(x_mask):
    r = np.array([1.0, 1.0, 1.0])
    x_binary = binarize_masks(x_mask)

    label = np.array([np.abs(np.sign(np.dot(i, 1))) for i in x_binary])
    mask = np.array([np.abs(np.sign(np.dot(i, r))) for i in x_mask])

    return label, mask


def extract_x_y_mask(x, paths, is_train_set=False):
    x_train, y_mask = files_to_numpy(
        x, (paths.dataset, '.jpg'), (paths.labels, '.png'), is_train_set
    )
    y_train, mask_train = labelise(y_mask)

    return x_train, y_train, mask_train


if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        exit(1)
        pass

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default='../dataset')
    parser.add_argument('--labels', default='../labels')
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

    # x_train, y_train, mask_train = extract_x_y_mask(_train_split_names, args)
    # x_val, y_val, mask_val = extract_x_y_mask(_train_split_names, args)

    sets = [
        extract_x_y_mask(_train_split_names, args, True),
        extract_x_y_mask(_val_split_names, args),
        extract_x_y_mask(_test_split_names, args)
    ]

    unet_main(sets)

    h = 0
