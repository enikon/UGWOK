import tensorflow as tf


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
        1, # num channels
        3, strides=2,
        padding='same'
    )  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def unet_main(sets):

    model = unet()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy', tf.keras.metrics.MeanIoU(1)]
    )

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir='../logs')

    model_history = model.fit(
        x=sets[0][0],
        y=sets[0][1],
        epochs=1,
        batch_size=20,
        validation_data=(sets[1][0], sets[1][1]),
        callbacks=[tb_cb]
    )

    pass

