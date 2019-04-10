#!/bin/bash

# python run.py --gpu="1" --lr=0.01 --epochs=30 --train_v="googlenet-2.0" --load_v="googlenet-2.0" --with_bn=1 --regularize=0 --retrain=0
# python run.py --gpu="1" --lr=0.001 --epochs=20 --train_v="googlenet-2.1" --load_v="googlenet-2.0" --with_bn=1 --regularize=0 --retrain=0

# python run.py --gpu="1" --lr=0.01 --epochs=15 --train_v="googlenet-3.0" --load_v="googlenet-3.0" --with_bn=1 --regularize=1 --retrain=0
# python run.py --gpu="1" --lr=0.001 --epochs=15 --train_v="googlenet-3.1" --load_v="googlenet-3.0" --with_bn=1 --regularize=1 --retrain=0

# python run.py --gpu="1" --lr=0.01 --epochs=15 --train_v="resnet-huge-1.0" --load_v="resnet-huge-1.0" --with_bn=1 --regularize=1 --retrain=0

python run.py --gpu="0" --lr=0.1 --epochs=20 --train_v="resnet-huge-1.0" --load_v="resnet-huge-1.0" --with_bn=1 --regularize=0 --retrain=0 --batch_size=64
