import numpy as np
import cv2
import albumentations as albu
from common import NUMBER_OF_CHANNELS, IMAGE_DIM_SIZE, AUGMENTATION_MULTIPLIER
import os
import datetime


def read_images(imges_names, path, dim_size):
    return np.array([cv2.resize(
        cv2.imread(os.path.join(path[0], i + path[1])),
        dim_size
    ) for i in imges_names])


def augment_data(images, masks, times=1):
    resultImages = []
    resultMasks = []

    transform_train = albu.Compose([
        albu.Flip(),
        albu.RandomRotate90(),
        albu.RandomResizedCrop(height=IMAGE_DIM_SIZE, width=IMAGE_DIM_SIZE, scale=(0.5, 1.0), p=0.5)
    ])

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
        resultImages = realImages
        resultMasks = realMasks

    return [resultImages, resultMasks]


def labelise(x_mask):
    r = np.array([1.0, 1.0, 1.0])
    x_binary = binarize_masks(x_mask)

    if NUMBER_OF_CHANNELS >= 2:
        label = np.array([np.abs(np.sign(np.dot(i, 1))) for i in x_binary], axis=-1)
        mask = np.array([np.abs(np.sign(np.dot(i, r))) for i in x_mask], axis=-1)
    else:
        label = np.expand_dims([np.abs(np.sign(np.dot(i, 1))) for i in x_binary], axis=-1)
        mask = np.expand_dims([np.abs(np.sign(np.dot(i, r))) for i in x_mask], axis=-1)

    return label, mask


def extract_x_y_mask(x, paths, is_train_set=False):
    x_train, y_mask = files_to_numpy(
        x, (paths.dataset, '.jpg'), (paths.labels, '.png'), is_train_set
    )
    y_train, mask_train = labelise(y_mask)
    return x_train, y_train, mask_train


def save_split(x, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        np.savetxt(f, x, fmt='%s', delimiter='\n')


def load_split(path):
    with open(path, 'rb') as f:
        x = np.loadtxt(f, dtype=str, delimiter='\n')
    return x


def save_image(image_name, image):
    cv2.imwrite(image_name + '.jpg', image)


# def convert_mask_to_pix(mask):
#     return np.array([list(map(lambda x: (0, 0, 128) if x == 1 else (0, 128, 128), row)) for row in mask])

def convert_mask_to_pix(mask, the_mask):
    return \
        np.expand_dims(mask, -1) * np.array([[[0, -128, 0]]]) + np.array([[[0, 128, 0]]])\
        + the_mask*np.array([[[0, 0, 64]]])


def save_masks_cmp(images, pred_masks, real_masks, the_mask, path):
    now = datetime.datetime.now()

    current_time = now.strftime("%Y%m%d-%H%M%S")
    saveFolder = os.path.join(path, 'predictions__' + current_time)
    os.makedirs(saveFolder, exist_ok=True)

    counter = 0
    for image, pred, real, th_mask in zip(images, pred_masks, real_masks, the_mask):
        save_image(saveFolder + '//img_' + str(counter), image)
        mask_pred_image = convert_mask_to_pix(pred, th_mask)
        save_image(saveFolder + '//img_' + str(counter) + '_pred', mask_pred_image)
        mask_image_real = convert_mask_to_pix(real, th_mask)
        save_image(saveFolder + '//img_' + str(counter) + '_real', mask_image_real)
        counter += 1

