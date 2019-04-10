import argparse
import os
import train
import test
import load_model
import my_GoogLeNet
import my_ResNet
import draw_his
import import_data


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store', type=str, default="0")
parser.add_argument('--lr', action='store', type=float, default=0.001)
parser.add_argument('--epochs', action='store', type=int, default=10)
parser.add_argument('--train_v', action='store', type=str, default="1.0")
parser.add_argument('--load_v', action='store', type=str, default="1.0")
parser.add_argument('--cifar_10_dir', action='store', type=str, default="/media/Data/datasets/cifar/cifar-10-python")
parser.add_argument('--load_data_dir', action='store', type=str, default="./data")
parser.add_argument('--with_bn', type=lambda x: bool(str2bool(x)), default=False)
parser.add_argument('--retrain', type=lambda x: bool(str2bool(x)), default=False)
parser.add_argument('--regularize', type=lambda x: bool(str2bool(x)), default=False)
parser.add_argument('--batch_size', action='store', type=int, default=128)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)


print("[info]: use gpu: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
print("[info]: set learning rate: {}".format(args.lr))
print("[info]: epochs: {}".format(args.epochs))
print("[info]: train_version: {}".format(args.train_v))
print("[info]: load_version: {}".format(args.load_v))
print("[info]: with bn: {}".format(args.with_bn))
print("[info]: retrain: {}".format(args.retrain))
print("[info]: regularize: {}".format(args.regularize))
print("[info]: batch_size: {}".format(args.batch_size))


model, create_new = load_model.load_model(args.load_v, my_ResNet.my_ResNet, args.retrain, args.regularize)
# model.summary()
# get data
train_set, valid_set, test_set, train_x_mean, train_x_std, label_names = import_data.import_data(
        cifar_10_dir="/media/Data/datasets/cifar/cifar-10-python",
        load_dir="./data",
        reload=False,
        valid_size=5000,
        to_BGR=True,
        to_channel_first=False
    )

train.train(
    model=model,
    train_version=args.train_v,
    train_set=train_set,
    valid_set=valid_set,
    lr=args.lr,
    epochs=args.epochs,
    batch_size=args.batch_size
)

test.test(
    test_version=args.train_v,
    test_set=test_set
)

draw_his.draw_his(args.train_v)
