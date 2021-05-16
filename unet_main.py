import os
from datetime import datetime
import math
import cv2

from metrics import *


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

    return tf.keras.Model(inputs=inputs, outputs=x)


def save_image(image_name, image):
    cv2.imwrite(image_name + '.jpg', image)


def create_mask(pred_mask):
    return np.argmax(pred_mask, axis=-1)


def convert_mask_to_pix(mask):
    return np.array([list(map(lambda x: (0, 0, 128) if x == 1 else (0, 128, 128), row)) for row in mask])


def add_sample_weights(label, weights):
    class_weights = tf.constant(weights)
    class_weights - class_weights / tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return sample_weights


def train_unet(sets):

    BATCH_SIZE = 4
    EPOCHS = 100

    model = unet()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        # FOR DEBUGGING
        # run_eagerly=True,
        metrics=[
            tf.keras.metrics.BinaryAccuracy() if NUMBER_OF_CHANNELS == 1 else tf.keras.metrics.Accuracy()
            , UpdatedThresholdMeanIoU(NUMBER_OF_CHANNELS, 0.5)
            , SteppedMeanIoU(num_classes=NUMBER_OF_CHANNELS, thresholds=THRESHOLDS)
            #, iou, iou_binary, mean_ap
        ]
    )

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir='../logs')
    m_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath='../models/last.h5', save_freq=math.ceil(sets[0][0].shape[0] / BATCH_SIZE), verbose=1)
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
        sample_weight=sets[2][2]
    )


def predict_unet(model, test_set):
    pred_pm = model.predict(x=test_set)
    pred_masks = create_mask(pred_pm)
    return pred_masks


def save_masks_cmp(images, pred_masks, real_masks):
    now = datetime.now()
    current_time = now.strftime("%Y%m%d-%H%M%S")
    saveFolder = 'predictions__' + current_time
    os.makedirs(saveFolder)

    counter = 0
    for image, pred, real in zip(images, pred_masks, real_masks):
        save_image(saveFolder + '//img_' + str(counter), image)
        mask_pred_image = convert_mask_to_pix(pred)
        save_image(saveFolder + '//img_' + str(counter) + '_pred', mask_pred_image)
        mask_image_real = convert_mask_to_pix(real)
        save_image(saveFolder + '//img_' + str(counter) + '_real', mask_image_real)
        counter += 1


