from keras import layers
from keras import models
from keras import regularizers
from keras.layers import Activation
from model_components import my_conv


def my_conv_for_inception(x, filters, kernel_size, use_regularizer):
    weight_decay = 0.0001
    x = my_conv(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            use_regularizer=use_regularizer,
            weight_decay=weight_decay
        )
    x = layers.BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def inception_1(x, channel_list, use_regularizer):
    # path 1
    x1 = my_conv_for_inception(x, channel_list['#1x1'], 1, use_regularizer)

    # path 2
    x2 = my_conv_for_inception(x, channel_list['#3x3 reduce'], 1, use_regularizer)
    x2 = my_conv_for_inception(x2, channel_list['#3x3'], 3, use_regularizer)

    # path 3
    x3 = my_conv_for_inception(x, channel_list['#5x5 reduce'], 1, use_regularizer)
    x3 = my_conv_for_inception(x3, channel_list['#5x5'], 5, use_regularizer)

    # path 4
    x4 = layers.MaxPool2D(3, 1, "same")(x)
    x4 = my_conv_for_inception(x4, channel_list['pool proj'], 1, use_regularizer)

    return layers.concatenate([x1, x2, x3, x4], axis=-1)


def my_inception(use_regularizer=True):
    print("[info]: googLeNet use regularizer: {}".format(use_regularizer))
    input_layer = layers.Input(
        shape=[32, 32, 3],
        name="inception_1_input"
    )

    x = my_conv_for_inception(input_layer, 32, 5, use_regularizer)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = my_conv_for_inception(x, 64, 3, use_regularizer)
    channel_list = {
        '#1x1': 32,
        '#3x3 reduce': 48,
        '#3x3': 64,
        '#5x5 reduce': 8,
        '#5x5': 16,
        'pool proj': 16
    }
    x = inception_1(x, channel_list, use_regularizer)
    channel_list = {
        '#1x1': 64,
        '#3x3 reduce': 64,
        '#3x3': 96,
        '#5x5 reduce': 16,
        '#5x5': 48,
        'pool proj': 32
    }
    x = inception_1(x, channel_list, use_regularizer)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    channel_list = {
        '#1x1': 96,
        '#3x3 reduce': 48,
        '#3x3': 104,
        '#5x5 reduce': 8,
        '#5x5': 24,
        'pool proj': 32
    }
    x = inception_1(x, channel_list, use_regularizer)
    x = inception_1(x, channel_list, use_regularizer)
    x = inception_1(x, channel_list, use_regularizer)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    channel_list = {
        '#1x1': 32,
        '#3x3 reduce': 48,
        '#3x3': 64,
        '#5x5 reduce': 8,
        '#5x5': 16,
        'pool proj': 16
    }
    x = inception_1(x, channel_list, use_regularizer)
    x = inception_1(x, channel_list, use_regularizer)

    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dropout(0.7)(x)

    regularizer = None
    if use_regularizer:
        regularizer = regularizers.l2(0.001)

    x = layers.Dense(10, activation='softmax', kernel_regularizer=regularizer)(x)

    model = models.Model(input_layer, x)
    return model


if __name__ == "__main__":
    import keras
    model = my_inception(False)
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.categorical_crossentropy
    )

    model.save("model/test_model.h5")


