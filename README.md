# Learning-Rate-Dropout
Pytorch implementation of Learning Rate Dropout.
Paper Link: https://arxiv.org/pdf/1912.00144.pdf

Train ResNet-34 for Cifar10:
run
    python main.py --model=resnet --optim=adam_lrd --lr=0.001 --LRD_p=0.5

    python main.py --model=resnet --optim=adam --lr=0.001

    python main.py --model=resnet --optim=sgd_lrd --lr=0.1 --LRD_p=0.5

    python main.py --model=resnet --optim=sgd --lr=0.1 --LRD_p=0.5

    python main.py --model=resnet --optim=rmsprop --lr=0.001

    python main.py --model=resnet --optim=rmsprop_lrd --lr=0.001 --LRD_p=0.5
    
After training, run "plot.py" to show the learning curves.
