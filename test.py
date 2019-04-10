import os
import keras
import load_model
import import_data
import my_ResNet


def test(test_version, test_set, batch_size=32):
    print("[info]: testing model...")
    # load model
    model, create_new = load_model.load_model(test_version, my_ResNet.my_ResNet, False)
    if create_new:
        print("[info]: try to test a non-trained model")
        exit(-1)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    test_log = model.evaluate(test_set[0], test_set[1], batch_size=batch_size)
    print("[info]: test loss: {:5f}, test acc: {:4f}".format(test_log[0], test_log[1]))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y), train_x_mean, train_x_std, label_names = \
        import_data.import_data(
            cifar_10_dir="/media/Data/datasets/cifar/cifar-10-python",
            load_dir="./data",
            reload=False,
            valid_size=5000,
            to_BGR=True,
            to_channel_first=False
        )
    test(
        test_version="resnet-1.2",
        test_set=(test_x, test_y),
        batch_size=128
    )
