"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from RMSprop import RMSprop
from rmsprop_lrd import RMSprop_LRD

from adam_lrd import Adam_LRD
from sgd_lrd import SGD_LRD


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet'])
    parser.add_argument('--optim', default='pid', type=str, help='optimizer',
                        choices=['sgd', 'adam','adam_lrd','sgd_lrd','rmsprop','rmsprop_lrd'])
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')

    parser.add_argument('--LRD_p', default=0.5, type=float, help='dropout_rate for LRD')
    return parser


def build_dataset():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                               num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1,momentum=0.9,
                  beta1=0.9, beta2=0.999, wd=1,LRD_p=0):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adam_lrd': 'lr{}-betas{}-{}-LRDp{}'.format(lr, beta1, beta2,LRD_p),
        'sgd_lrd': 'lr{}-momentum{}--LRDp{}'.format(lr, momentum,LRD_p),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'rmsprop': 'lr{}'.format(lr),
        'rmsprop_lrd': 'lr{}--LRDp{}'.format(lr, LRD_p),

    }[optimizer]
    return '{}-{}-{}'.format(model, optimizer, name)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(ckpt_name)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'sgd_lrd':
        return SGD_LRD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay,dropout=args.LRD_p)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'adam_lrd':
        return Adam_LRD(model_params, args.lr, betas=(args.beta1, args.beta2),weight_decay=args.weight_decay,dropout=args.LRD_p)
    elif args.optim == 'rmsprop':
        return RMSprop(model_params, args.lr, alpha=0.99,weight_decay=args.weight_decay, momentum=0)
    elif args.optim == 'rmsprop_lrd':
        return RMSprop_LRD(model_params, args.lr, alpha=0.99, weight_decay=args.weight_decay, momentum=0,
                          dropout=args.LRD_p)


def train(net, epoch, device, data_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # optimizer.step(epoch=epoch)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('train acc %.3f' % accuracy)
    print('train loss %.6f' % train_loss)

    return accuracy, train_loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(' test acc %.3f' % accuracy)

    return accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, wd=args.weight_decay,LRD_p=args.LRD_p)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    print(net.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1,
                                          last_epoch=start_epoch)

    train_accuracies = []
    test_accuracies = []
    train_loss_data = []

    for epoch in range(start_epoch + 1, 200):
        scheduler.step()
        # print("lr=", args.lr)
        train_acc, train_loss= train(net, epoch, device, train_loader, optimizer, criterion)
        test_acc = test(net, device, test_loader, criterion)

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        train_loss_data.append(train_loss)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'train_loss': train_loss_data, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))


if __name__ == '__main__':
    main()
