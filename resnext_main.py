import tensorflow as tf
import numpy as np

from common import NUMBER_OF_CHANNELS, IMAGE_DIM_SIZE, THRESHOLDS, BATCH_SIZE, EPOCHS
from metrics import UpdatedThresholdMeanIoU, SteppedMeanIoU
from resnext.resnext_block import build_ResNeXt_block


def resnext_layers():
    down = [
        build_ResNeXt_block(filters=128,
                            strides=1,
                            groups=32,
                            repeat_num=3),
        build_ResNeXt_block(filters=256,
                            strides=2,
                            groups=32,
                            repeat_num=4),
        build_ResNeXt_block(filters=512,
                            strides=2,
                            groups=32,
                            repeat_num=6)
    ]
    up = [
        build_ResNeXt_block(filters=512,
                            strides=2,
                            groups=32,
                            repeat_num=6),
        build_ResNeXt_block(filters=256,
                            strides=2,
                            groups=32,
                            repeat_num=4),
        build_ResNeXt_block(filters=128,
                            strides=1,
                            groups=32,
                            repeat_num=3)
    ]

    mid = build_ResNeXt_block(filters=1024,
                              strides=2,
                              groups=32,
                              repeat_num=3)

    return down, up, mid


def resnext_compile(model):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy() if NUMBER_OF_CHANNELS == 1 else tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy() if NUMBER_OF_CHANNELS == 1 else 'accuracy'
            , UpdatedThresholdMeanIoU(NUMBER_OF_CHANNELS, 0.5)
            , SteppedMeanIoU(num_classes=NUMBER_OF_CHANNELS, thresholds=THRESHOLDS)
            # , iou, iou_binary, mean_ap
        ])


def resnext():
    inputs = tf.keras.layers.Input(shape=[IMAGE_DIM_SIZE, IMAGE_DIM_SIZE, 3])
    conv = tf.keras.layers.Conv2D(48, kernel_size=(3, 3))(inputs)
    x = conv

    down, up, mid = resnext_layers()
    xout = [None for i in range(len(down))]

    for i, down_i in enumerate(down):
        xout[i] = down_i(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = mid(x)

    embedding = x
    xout = reversed(xout[:-1])

    for up_i, xo in zip(up, xout):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, xo])
        x = up_i(x)

    last = tf.keras.layers.Conv2DTranspose(
        NUMBER_OF_CHANNELS,
        3, strides=2,
        padding='same'
    )

    x = last(x)

    if NUMBER_OF_CHANNELS == 1:
        x = tf.keras.activations.sigmoid(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    resnext_compile(model)

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


def train_resnext(sets):
    model = resnext()

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir='../logs')
    m_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(filepath='../models/last.h5', save_freq='epoch', verbose=1)
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


def evaluate_resnext(model, sets):
    print("\nEVALUATION")
    model.evaluate(
        x=sets[2][0],
        y=sets[2][1],
        sample_weight=sets[2][2],
        batch_size=BATCH_SIZE
    )


def predict_resnext(model, test_set):
    pred_pm = model.predict(x=test_set, batch_size=BATCH_SIZE)
    pred_masks = create_mask(pred_pm)
    return pred_masks
