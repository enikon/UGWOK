import os
import tensorflow as tf
import argparse

from unet_main import unet_compile, evaluate_unet, predict_unet
from datasets import extract_x_y_mask, load_split, save_masks_cmp

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default='../dataset')
    parser.add_argument('--labels', default='../labels')
    parser.add_argument('--split_dir', default='../split') #filenames

    args = parser.parse_args()
    _test_split_names = load_split(os.path.join(args.split_dir, 'test'))

    sets = [
        None,
        None,
        extract_x_y_mask(_test_split_names, args)
    ]

    model = tf.keras.models.load_model(
        '../models/best.h5',
        compile=False
    )
    unet_compile(model)

    evaluate_unet(model, sets)
    pred_masks = predict_unet(model, sets[2][0])
    save_masks_cmp(sets[2][0], pred_masks, sets[2][1], '../results')

    h = 0
