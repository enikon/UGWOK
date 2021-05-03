import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


def unet_pretrained_encoder():
    # shamelessly stolen from:
    # https://www.tensorflow.org/tutorials/images/segmentation?fbclid=IwAR03fzGTYWlyGqF-Ht7dz8ckslyFXe-ZSJgT2gZGASESpIN0GBymU3DFjP4

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[128, 128, 3],
        input_tensor=tf.keras.layers.Input(shape=[128, 128, 3]),
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
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

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
        2, # num channels
        3, strides=2,
        padding='same'
    )  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# TODO: prediction part, later move to prediction function(model, testSet)  or module         START
def saveImage(imageName, image):
    cv2.imwrite(imageName + '.jpg', image)


def create_mask(pred_mask):
    return np.argmax(pred_mask, axis=-1)


def convert_mask_to_pix(mask):
    return np.array([list(map(lambda x: (0, 0, 128) if x == 1 else (0, 128, 128), row)) for row in mask])
# TODO: prediction part, later move to prediction function(model, testSet)  or module         END


def unet_main(sets):

    model = unet()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir='../logs')

    model_history = model.fit(
        x=sets[0][0],
        y=sets[0][1],
        epochs=20,
        batch_size=16,
        validation_data=(sets[1][0], sets[1][1]),
        callbacks=[tb_cb]
    )

# TODO: prediction part, later move to prediction function(model, testSet) or module

    pred_pm = model.predict(
        x=sets[2][0]
    )

    pred_masks = create_mask(pred_pm)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    saveFolder = 'predictions__' + current_time
    os.makedirs(saveFolder)

    counter = 0
    for pred, real in zip(pred_pm, sets[2][1]):
        mask_pred_image = convert_mask_to_pix(create_mask(pred))
        # plt.imshow(mask_pred_image) nie działa
        saveImage(saveFolder + '//img_' + str(counter) + '_pred', mask_pred_image)
        mask_image_real = convert_mask_to_pix(real)
        # plt.imshow(mask_image_real) nie działa
        saveImage(saveFolder + '//img_' + str(counter) + '_real', mask_image_real)
        counter += 1

    pass

