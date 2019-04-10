import keras
import os


def load_model(version, new_model, retrain=False, *args):
    """
    :param version: model version
    :param new_model: method for call to get a new model e.g. my_ResNet.my_ResNet
    :param retrain: True: load new model
    :return:
    """
    create_new_model = False
    # load model
    if not retrain:
        try:
            with open(os.path.join("./model", "model_structure_{}.json".format(version)), "r") as file:
                model_json = file.read()
            print("[info]: loading model...")
            model = keras.models.model_from_json(model_json)
            model.load_weights(os.path.join("./model", "model_weights_{}.h5".format(version)))
            print("[info]: load model done.")
        except OSError:
            print("[info]: load model file failed, creating model")
            model = new_model(*args)
            create_new_model = True
    else:
        print("[info]: retrain, creating model")
        model = new_model(*args)
        create_new_model = True
    return model, create_new_model

