Cifar-10:

python main.py --model=resnet --optim=adam_lrd --lr=0.001 --LRD_p=0.5

python main.py --model=resnet --optim=adam --lr=0.001

python main.py --model=resnet --optim=sgd_lrd --lr=0.1 --LRD_p=0.5

python main.py --model=resnet --optim=sgd --lr=0.1 --LRD_p=0.5

python main.py --model=resnet --optim=rmsprop --lr=0.001

python main.py --model=resnet --optim=rmsprop_lrd --lr=0.001 --LRD_p=0.5