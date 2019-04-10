import numpy as np
import os
import pickle
import parms
from parms import train_batch_num
import data_augmentation


def get_raw_data(cifar_10_dir):
    """
    cifar_10_dir: cifar-10 dataset dir
    return: train_x, train_y, test_x, test_y, label_names (train_x and test_x are in RGB channel last)
    """
    # load meta data
    meta_data_dir = os.path.join(cifar_10_dir, parms.meta_file_name)
    with open(meta_data_dir, "rb") as meta_file:
        meta_dict = pickle.load(meta_file, encoding='bytes')
    batch_size = meta_dict[b'num_cases_per_batch']  # 10000
    data_len = meta_dict[b'num_vis']  # 3072
    label_names = meta_dict[b'label_names']
    label_names = list(map(lambda x: x.decode('utf-8'), label_names))
    meta_dict.clear()

    # get train dataset
    # 5 train batch in total
    train_x = np.zeros([train_batch_num * batch_size, data_len]).astype(np.uint8)
    train_label = np.zeros([train_batch_num * batch_size]).astype(np.uint8)
    for bn in range(train_batch_num):
        train_file_dir = os.path.join(cifar_10_dir, parms.train_file_name_prefix.format(bn + 1))
        with open(train_file_dir, "rb") as train_file:
            train_dict = pickle.load(train_file, encoding='bytes')
        print("load " + train_dict[b'batch_label'].decode('utf-8'))
        train_x[batch_size * bn:batch_size * (bn + 1)] = np.array(train_dict[b'data'])
        train_label[batch_size * bn:batch_size * (bn + 1)] = np.array(train_dict[b'labels'])
        train_dict.clear()  # release memory
    train_x = train_x.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])

    # get test dataset
    test_file_dir = os.path.join(cifar_10_dir, parms.test_file_name)
    with open(test_file_dir, "rb") as test_file:
        test_dict = pickle.load(test_file, encoding='bytes')
    print("load " + test_dict[b'batch_label'].decode('utf-8'))
    test_x = np.array(test_dict[b'data']).astype(np.uint8)
    test_label = np.array(test_dict[b'labels']).astype(np.uint8)
    test_dict.clear()  # release memory
    test_x = test_x.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])

    return train_x, train_label, test_x, test_label, label_names


def data_preprocess(
        train_x, train_label, test_x, test_label, class_num,
        to_BGR, to_channel_first, shuffle=True
):
    """
    to_BGR: to_BGR==True: convert x to BGR mode, or keep x as RGB mode
    to_channel_first: to_channel_first==True: convert x to channel first mode which is recommended for GPU
        computation; to_channel_first==False is recommended fot CPU computation
    return: normalized train_x, test_x normalized with means of train_x and one-hot y and train_x means based on channel
    """
    # shuffle
    if shuffle:
        # shuffle data
        index = np.linspace(0, train_x.shape[0] - 1, train_x.shape[0]).astype(np.int)
        np.random.shuffle(index)
        train_x = train_x[index]
        train_label = train_label[index]

    # to BGR
    if to_BGR:
        train_x = train_x[..., ::-1]
        test_x = test_x[..., ::-1]

    # to channel first
    if to_channel_first:
        train_x = train_x.transpose([0, 3, 1, 2])
        test_x = test_x.transpose([0, 3, 1, 2])

    # convert label to one hot code
    train_y = np.eye(train_label.shape[0], class_num)[train_label].astype(np.float32)
    test_y = np.eye(test_label.shape[0], class_num)[test_label].astype(np.float32)

    return train_x, train_y, test_x, test_y


def import_data(
        cifar_10_dir, load_dir, reload=False, valid_size=5000,
        to_BGR=True, to_channel_first=False
):
    """
    :param cifar_10_dir:
    :param reload: if reload==True, reload whole dataset and do preprocess, else load preprocessed data from disk
    :param valid_size: size of valid set
    :return: train_x, train_y, test_x, test_y, train_x_mean, label_names
    """
    if not reload:
        try:
            train_x = np.load(os.path.join(load_dir, "train_x.npy"))
            train_y = np.load(os.path.join(load_dir, "train_y.npy"))
            valid_x = np.load(os.path.join(load_dir, "valid_x.npy"))
            valid_y = np.load(os.path.join(load_dir, "valid_y.npy"))
            test_x = np.load(os.path.join(load_dir, "test_x.npy"))
            test_y = np.load(os.path.join(load_dir, "test_y.npy"))
            train_x_mean = np.load(os.path.join(load_dir, "train_x_mean.npy"))
            train_x_std = np.load(os.path.join(load_dir, "train_x_std.npy"))
            label_names = np.load(os.path.join(load_dir, "label_names.pkl"))

            return (train_x, train_y), (valid_x, valid_y), (test_x, test_y), train_x_mean, train_x_std, label_names
        except FileNotFoundError:
            print("file not found, reload whole dataset")
    # get raw data
    raw_train_x, raw_train_label, raw_test_x, raw_test_label, label_names = get_raw_data(cifar_10_dir)

    # data preprocess
    train_x, train_y, test_x, test_y = data_preprocess(
        raw_train_x, raw_train_label, raw_test_x, raw_test_label, len(label_names),
        to_BGR=to_BGR,
        to_channel_first=to_channel_first,
        shuffle=True
    )

    # split valid set
    valid_x, train_x = train_x[0:valid_size, ...], train_x[valid_size:, ...]
    valid_y, train_y = train_y[0:valid_size, ...], train_y[valid_size:, ...]

    # data augmentation
    train_x, train_y = data_augmentation.data_augmentation(
        train_x,
        train_y,
        data_format="channel_last"
    )
    # data normalization
    if to_channel_first:
        train_x_mean = np.mean(train_x, axis=(0, 2, 3)).astype(np.float32)
    else:
        train_x_mean = np.mean(train_x, axis=(0, 1, 2)).astype(np.float32)

    train_x_std = np.std(train_x)
    train_x = (train_x - train_x_mean) / train_x_std
    valid_x = (valid_x - train_x_mean) / train_x_std
    test_x = (test_x - train_x_mean) / train_x_std

    print("[info]: mean: {}, std: {}".format(train_x_mean, train_x_std))

    # save dataset
    print("writing train_x...")
    np.save(os.path.join(load_dir, "train_x.npy"), train_x)
    print("writing train_y...")
    np.save(os.path.join(load_dir, "train_y.npy"), train_y)
    print("writing valid_x...")
    np.save(os.path.join(load_dir, "valid_x.npy"), valid_x)
    print("writing valid_y...")
    np.save(os.path.join(load_dir, "valid_y.npy"), valid_y)
    print("writing test_x...")
    np.save(os.path.join(load_dir, "test_x.npy"), test_x)
    print("writing test_y...")
    np.save(os.path.join(load_dir, "test_y.npy"), test_y)
    print("writing train_x_mean...")
    np.save(os.path.join(load_dir, "train_x_mean.npy"), train_x_mean)
    print("writing train_x_std...")
    np.save(os.path.join(load_dir, "train_x_std.npy"), train_x_std)
    print("writing label_names...")
    with open(os.path.join(load_dir, "label_names.pkl"), "wb") as file:
        pickle.dump(label_names, file)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y), train_x_mean, train_x_std, label_names


def test(cifar_10_dir, load_dir):
    import cv2
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y), train_x_mean, train_x_std, label_names = \
        import_data(cifar_10_dir, load_dir, reload=True, valid_size=5000, to_channel_first=False)
    # print(label_names)
    # test

    # label_file = open("label_record.txt", "w")
    # print(valid_x.shape, train_y.shape, train_x.dtype)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    for i in range(20):
        index = np.random.random_integers(0, valid_x.shape[0] - 1)
        x = valid_x[index]    # channel last
        x = x*train_x_std + train_x_mean
        x[x < 0] = 0
        x[x > 255] = 255
        x = x.astype(np.uint8)

        # cv2.imwrite("{}.png".format(i), x)
        # label_file.write(label_names[train_y[index].argmax()] + "\n")
        cv2.imshow("test", x)
        print(label_names[valid_y[index].argmax()])
        cv2.waitKey()
    # label_file.close()


if __name__ == "__main__":
    test("/media/Data/datasets/cifar/cifar-10-python", "./data")


