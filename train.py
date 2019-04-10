import keras
import os
import numpy as np
import import_data


def fit(model, train_set, valid_set, batch_size, epochs, train_version, freq):
    """
    freq: freq (in step) of recording train history
    """
    # train
    train_steps = int(train_set[0].shape[0]/batch_size)
    last_step_batch_size = train_set[0].shape[0] - train_steps*batch_size
    print("[info]: train on {}, valid on {}, last step batch size: {}".
          format(train_set[0].shape[0], valid_set[0].shape[0], last_step_batch_size))

    # train history
    loss = np.array(list()).astype(np.float32)
    acc = np.array(list()).astype(np.float32)
    loss_val = np.array(list()).astype(np.float32)
    acc_val = np.array(list()).astype(np.float32)

    for epoch in range(epochs):
        best_loss = np.inf
        best_acc = 0
        for step in range(train_steps):
            x = train_set[0][step*batch_size:(step + 1)*batch_size]
            y = train_set[1][step*batch_size:(step + 1)*batch_size]
            metrics = model.train_on_batch(x, y)
            if step % freq == 0:
                loss = np.append(loss, metrics[0])
                acc = np.append(acc, metrics[1])

                if metrics[0] < best_loss and metrics[1] > best_acc:
                    best_loss = metrics[0]
                    best_acc = metrics[1]
                    model.save_weights(
                        os.path.join("./model", "model_weights_{}_epoch_{}.h5".format(train_version, epoch)))
                    print("\n[info]: save model with loss: {:.5f}, acc: {:.4f}".format(best_loss, best_acc), end="\n")
                # pred = model.predict(x)
                # pred[pred < 1e-07] = 1e-7
                # pred_loss = -np.sum(y*np.log(pred), axis=1).mean()
                # print("\n[info]: pure pred loss: {}".format(pred_loss))

            print("\repoch: {}/{}, step: {}/{}, loss: {:.5f}, acc: {:.4f}".
                  format(epoch + 1, epochs, step, train_steps, metrics[0], metrics[1]), end="")

        # left data
        if last_step_batch_size != 0:
            x = train_set[0][train_steps * batch_size:]
            y = train_set[1][train_steps * batch_size:]
            model.train_on_batch(x, y)

        # valid
        print("\n[info]: validate on epoch {} final model".format(epoch))
        valid_log = model.evaluate(valid_set[0], valid_set[1], batch_size=batch_size)
        loss_val = np.append(loss_val, valid_log[0])
        acc_val = np.append(acc_val, valid_log[1])
        model.save_weights(
            os.path.join("./model", "model_weights_{}_epoch_{}_final.h5".format(train_version, epoch))
        )
        print("[info]: val loss: {:.5f}, val acc: {:.4f}".format(valid_log[0], valid_log[1]))
    # after training
    np.save(os.path.join("./logs", "loss_his_{}".format(train_version)), loss)
    np.save(os.path.join("./logs", "acc_his_{}".format(train_version)), acc)
    np.save(os.path.join("./logs", "loss_val_his_{}".format(train_version)), loss_val)
    np.save(os.path.join("./logs", "acc_val_his_{}".format(train_version)), acc_val)
    return loss_val, acc_val


def train(model, train_version, train_set, valid_set, lr, epochs=10, watch_freq=10, batch_size=128):

    # compile model
    my_optimizer = keras.optimizers.Adagrad(lr)
    loss = keras.losses.categorical_crossentropy
    model.compile(
        optimizer=my_optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    # write model structure
    json_string = model.to_json()
    with open(os.path.join("./model", "model_structure_{}.json".format(train_version)), "w") as file:
        file.write(json_string)

    # fit
    loss_val, acc_val = fit(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        batch_size=batch_size,
        epochs=epochs,
        train_version=train_version,
        freq=watch_freq
    )
    # get best model
    tmp_model = keras.models.model_from_json(json_string)
    tmp_model.compile(
        optimizer=my_optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    print("[info]: getting best model...")
    # best final acc
    best_final_id = np.argmax(acc_val)
    print("[info]: best final id: {}, val acc: {:.4f}".format(best_final_id, acc_val[best_final_id]))
    # get best epoch model
    best_epoch_acc_val = np.array(list())
    best_epoch_loss_val = np.array(list())
    for ep in range(epochs):
        tmp_model.load_weights(
            os.path.join("./model", "model_weights_{}_epoch_{}.h5".format(train_version, ep))
        )
        log = tmp_model.evaluate(valid_set[0], valid_set[1], batch_size=64)
        print("[info]: epoch {}: val acc: {:.4f}".format(ep, log[1]))
        best_epoch_loss_val = np.append(best_epoch_loss_val, log[0])
        best_epoch_acc_val = np.append(best_epoch_acc_val, log[1])
    best_epoch_id = np.argmax(best_epoch_acc_val)
    print("[info]: best epoch val acc: {:.4f}".format(best_epoch_acc_val[best_epoch_id]))
    if acc_val[best_final_id] > best_epoch_acc_val[best_epoch_id]:
        print("[info]: choose batch final module")
        tmp_model.load_weights(
            os.path.join("./model", "model_weights_{}_epoch_{}_final.h5".format(train_version, best_final_id))
        )
    else:
        print("[info]: choose batch best module")
        tmp_model.load_weights(
            os.path.join("./model", "model_weights_{}_epoch_{}.h5".format(train_version, best_epoch_id))
        )
    tmp_model.save_weights(
        os.path.join("./model", "model_weights_{}.h5".format(train_version))
    )


def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    import my_ResNet
    import load_model
    my_model, _ = load_model.load_model("test", my_ResNet.my_ResNet, True, False)
    train_set, valid_set, test_set, _, _ = import_data.import_data(
        cifar_10_dir="/media/Data/datasets/cifar/cifar-10-python",
        load_dir="./data",
        reload=False,
        valid_size=5000,
        to_BGR=True,
        to_channel_first=False
    )
    train(
        model=my_model,
        train_version="test",
        train_set=train_set,
        valid_set=valid_set,
        lr=1e-3,
        epochs=10,
        batch_size=32
    )


if __name__ == "__main__":
    test()

