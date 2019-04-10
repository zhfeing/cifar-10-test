import keras
from keras import layers
import model_components


def res_block(x, filters, use_regularizer, with_subsample=False, weight_decay=0.0001):
    """
    :param x:
    :param filters:
    :param with_bypass:
    :param regularizer: regularizer
    :return:
    """
    if with_subsample:
        x_1 = model_components.my_conv_with_bn_relu(
            x=x,
            filters=filters,
            kernel_size=3,
            padding="same",
            stride=2,
            use_regularizer=use_regularizer,
            weight_decay=weight_decay
        )

    else:
        x_1 = model_components.my_conv_with_bn_relu(
            x=x,
            filters=filters,
            kernel_size=3,
            padding="same",
            stride=1,
            use_regularizer=use_regularizer,
            weight_decay=weight_decay
        )

    x_1 = model_components.my_conv(
            x=x_1,
            filters=filters,
            kernel_size=3,
            padding="same",
            stride=1,
            use_regularizer=use_regularizer,
            weight_decay=weight_decay
        )
    x_1 = layers.BatchNormalization()(x_1)

    if with_subsample:
        shortcut = model_components.my_conv(
            x=x,
            filters=filters,
            kernel_size=1,
            padding="same",
            stride=2,
            use_regularizer=use_regularizer,
            weight_decay=weight_decay
        )
    else:
        shortcut = x

    x = layers.Add()([shortcut, x_1])
    x = layers.Activation('relu')(x)
    return x


def my_ResNet(use_regularizer=True):
    print("[info]: use regularizer: {}".format(use_regularizer))

    weight_decay = 0.0001

    input_tensor = layers.Input(shape=[32, 32, 3])

    x = model_components.my_conv_with_bn_relu(
            x=input_tensor,
            filters=16,
            kernel_size=3,
            padding="same",
            stride=1,
            use_regularizer=use_regularizer,
            weight_decay=weight_decay
        )

    x = res_block(x, 16, use_regularizer, weight_decay=weight_decay)
    x = res_block(x, 16, use_regularizer, weight_decay=weight_decay)
    x = res_block(x, 16, use_regularizer, weight_decay=weight_decay)

    x = res_block(x, 32, use_regularizer, with_subsample=True, weight_decay=weight_decay)
    x = res_block(x, 32, use_regularizer, weight_decay=weight_decay)
    x = res_block(x, 32, use_regularizer, weight_decay=weight_decay)

    x = res_block(x, 64, use_regularizer, with_subsample=True, weight_decay=weight_decay)
    x = res_block(x, 64, use_regularizer, weight_decay=weight_decay)
    x = res_block(x, 64, use_regularizer, weight_decay=weight_decay)

    x = layers.GlobalAvgPool2D()(x)
    # x = layers.Dropout(0.7)(x)

    regularizer = None
    if use_regularizer:
        regularizer = keras.regularizers.l2(0.0001)

    x = layers.Dense(
        10,
        activation='softmax',
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer
    )(x)

    model = keras.Model(input_tensor, x)

    return model


if __name__ == "__main__":
    my_res_net = my_ResNet()
    my_res_net.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.categorical_crossentropy,
        metrics=['acc']
    )
    my_res_net.summary()
    my_res_net.save("./model/test_model.h5")


