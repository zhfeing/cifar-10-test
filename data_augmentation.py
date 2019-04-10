import numpy as np
import cv2


def data_augmentation(train_x, train_y, data_format="channel_last"):
    """
    data_format: only implemented with channel_last
    """
    if data_format != "channel_last":
        print("[info] only implemented with channel last data format")
        exit(-1)
    x_shape = train_x.shape
    y_shape = train_y.shape
    # padding and randomly flip
    padding_size = 4
    flip_pr = 0.5       # probability on flipping the img horizontally
    flipped_img = []
    flipped_label = []

    for i in range(train_x.shape[0]):
        img = train_x[i]
        # randomly flip img
        flip = np.random.binomial(1, flip_pr)
        if flip == 1:       # flip this img
            img = cv2.flip(img, 1)

        img = cv2.copyMakeBorder(
            src=img,
            top=padding_size, bottom=padding_size, left=padding_size, right=padding_size,
            borderType=cv2.BORDER_REPLICATE
        )
        o_x = np.random.randint(0, 2 * padding_size)
        o_y = np.random.randint(0, 2 * padding_size)
        img = img[o_y:o_y + x_shape[1], o_x:o_x + x_shape[2]]

        # write back
        if flip == 1:
            # flipped_img.append(img)
            # flipped_label.append(train_y[i])
            train_x[i] = img
        else:
            train_x[i] = img

    # flipped_img = np.array(flipped_img)
    # flipped_label = np.array(flipped_label)

    # append flipped imgs
    # train_x = np.append(train_x, flipped_img, axis=0)
    # train_y = np.append(train_y, flipped_label, axis=0)

    # add noise
    noise = np.random.randn(*train_x.shape).astype(np.float32)*1.414
    print("[info] noise average: {}, noise var: {}, noise max: {}".format(np.mean(noise), np.var(noise), np.max(noise)))
    train_x = train_x.astype(np.float32) + noise
    train_x[train_x < 0] = 0
    train_x[train_x > 255] = 255
    train_x = train_x.astype(np.uint8)

    return train_x, train_y


def test():
    import import_data
    train_x, train_label, test_x, test_label, label_names = \
        import_data.get_raw_data("/media/Data/datasets/cifar/cifar-10-python")
    train_x, train_y, test_x, test_y = \
        import_data.data_preprocess(train_x, train_label, test_x, test_label, len(label_names), True, False)
    train_x, train_y = data_augmentation(train_x, train_y)

    print(train_x.shape, train_y.shape)

    for i in range(100):
        img = train_x[i]
        label = label_names[np.argmax(train_y[i])]

        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.imshow(label, img)
        cv2.waitKey()
        cv2.destroyWindow(label)


if __name__ == "__main__":
    test()










