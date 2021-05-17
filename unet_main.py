import tensorflow as tf
import math
import cv2
import numpy as np

from common import NUMBER_OF_CHANNELS, IMAGE_DIM_SIZE, THRESHOLDS, BATCH_SIZE, EPOCHS
from metrics import UpdatedThresholdMeanIoU, SteppedMeanIoU


def unet_compile(model):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy() if NUMBER_OF_CHANNELS == 1 else tf.keras.metrics.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy() if NUMBER_OF_CHANNELS == 1 else tf.keras.metrics.Accuracy()
            , UpdatedThresholdMeanIoU(NUMBER_OF_CHANNELS, 0.5)
            , SteppedMeanIoU(num_classes=NUMBER_OF_CHANNELS, thresholds=THRESHOLDS)
            # , iou, iou_binary, mean_ap
        ])


def unet_pretrained_encoder():
    # shamelessly stolen from:
    # https://www.tensorflow.org/tutorials/images/segmentation?fbclid=IwAR03fzGTYWlyGqF-Ht7dz8ckslyFXe-ZSJgT2gZGASESpIN0GBymU3DFjP4

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[IMAGE_DIM_SIZE, IMAGE_DIM_SIZE, 3],
        input_tensor=tf.keras.layers.Input(shape=[IMAGE_DIM_SIZE, IMAGE_DIM_SIZE, 3]),
        include_top=False)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    # Needed for skip-connections
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    # Disable automatic backpropagation training
    # For inference only
    down_stack.trainable = False
    return down_stack


def unet():
    inputs = tf.keras.layers.Input(shape=[IMAGE_DIM_SIZE, IMAGE_DIM_SIZE, 3])

    up_stack_layers = [
        tf.keras.layers.UpSampling2D((2, 2)),  # 4x4 -> 8x8
        tf.keras.layers.UpSampling2D((2, 2)),  # 8x8 -> 16x16
        tf.keras.layers.UpSampling2D((2, 2)),  # 16x16 -> 32x32
        tf.keras.layers.UpSampling2D((2, 2)),  # 32x32 -> 64x64
    ]

    # Downsampling through the model
    skips = unet_pretrained_encoder()(inputs)

    embedding = skips[-1]
    skips = reversed(skips[:-1])

    x = embedding
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack_layers, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        NUMBER_OF_CHANNELS,
        3, strides=2,
        padding='same'
    )  # 64x64 -> 128x128

    x = last(x)
    if NUMBER_OF_CHANNELS >= 2:
        x = tf.keras.activations.softmax(x)
    else:
        x = tf.keras.activations.sigmoid(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    unet_compile(model)

    return model


def create_mask(pred_mask):
    if NUMBER_OF_CHANNELS >= 2:
        x = np.argmax(pred_mask, axis=-1)
    else:
        x = np.squeeze(np.rint(pred_mask))
    return x


def add_sample_weights(label, weights):
    class_weights = tf.constant(weights)
    class_weights - class_weights / tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return sample_weights


def train_unet(sets):

    model = unet()

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir='../logs')
    m_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath='../models/last.h5',
                                                   save_freq=math.ceil(sets[0][0].shape[0] / BATCH_SIZE), verbose=1)
    m_best_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath='../models/best.h5', save_best_only=True, verbose=1)

    # sample_weights = add_sample_weights(sets[0][1], [1.0, 1.5])
    sample_weights = sets[0][2]

    model_history = model.fit(
        x=sets[0][0],
        y=sets[0][1],
        sample_weight=sample_weights,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(sets[1][0], sets[1][1], sets[1][2]),
        callbacks=[tb_cb, m_ckpt_cb, m_best_ckpt_cb]
    )
    return model, model_history


def evaluate_unet(model, sets):
    print("\nEVALUATION")
    model.evaluate(
        x=sets[2][0],
        y=sets[2][1],
        sample_weight=sets[2][2],
        batch_size=BATCH_SIZE
    )


def predict_unet(model, test_set):
    pred_pm = model.predict(x=test_set, batch_size=BATCH_SIZE)
    pred_masks = create_mask(pred_pm)
    return pred_masks
