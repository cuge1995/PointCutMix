from __future__ import print_function
import argparse
import os
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_utils.data_util import PointcloudScaleAndTranslate
from data_utils.ModelNetDataLoader import ModelNetDataLoader

from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2 import PointNet2ClsMsg
from models.dgcnn import DGCNN
from models.pointcnn import PointCNNCls

from utils import progress_bar, log_row
import sys
sys.path.append("./emd/")
import emd_module as emd


def gen_train_log(args):
    if not os.path.isdir('logs_train'):
        os.mkdir('logs_train')
    logname = ('logs_train/%s_%s_%s.csv' % (args.data, args.model, args.name))

    if os.path.exists(logname):
        with open(logname, 'a') as logfile:
            log_row(logname, [''])
            log_row(logname, [''])

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['model type', 'data set', 'seed', 'train batch size',
                            'number of points in one batch', 'number of epochs', 'optimizer',
                            'learning rate', 'resume checkpoint path',
                            'feature transform', 'lambda for feature transform regularizer', 'data augment'])
        logwriter.writerow([args.model, args.data, args.seed, args.batch_size, args.num_points,
                            args.epochs, args.optimizer, args.lr, args.resume,
                            args.feature_transform, args.lambda_ft, args.augment])
        logwriter.writerow(['Note', args.note])
        logwriter.writerow([''])


def save_ckpt(args, epoch, model, optimizer, acc_list):
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir('checkpoints/%s_%s_%s' % (args.data, args.model, args.name)):
        os.mkdir('checkpoints/%s_%s_%s' % (args.data, args.model, args.name))
    if acc_list[-1] > max(acc_list[:-1]):
        print('=====> Saving checkpoint...')
        print('the best test acc is', acc_list[-1])
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'acc_list': acc_list,
        }
        torch.save(state, 'checkpoints/%s_%s_%s/best.pth' % (args.data, args.model, args.name))
        print('Successfully save checkpoint at epoch %d' % epoch)


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def test(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader, 0):
        points, label = data
        points, label = points.to(device), label.to(device)[:, 0]

        if args.model == 'rscnn_rcutmix':
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                              2).contiguous()  # (B, N, 3)
        points = points.transpose(2, 1)  # to be shape batch_size*3*N

        pred, trans_feat = model(points)

        loss = criterion(pred, label.long())

        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(label.data).cpu().sum()
        total += label.size(0)
        progress_bar(j, len(test_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                     % (loss.item() / (j + 1), 100. * correct.item() / total, correct, total))

    return loss.item() / (j + 1), 100. * correct.item() / total


if __name__ == '__main__':
    ########################################
    ## Set hypeparameters
    ########################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointnet', help='choose model type')
    parser.add_argument('--data', type=str, default='modelnet40', help='choose data set')
    parser.add_argument('--seed', type=int, default=0, help='manual random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--resume', type=str, default='/', help='resume path')
    parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")
    parser.add_argument('--lambda_ft', type=float, default=0.001, help="lambda for feature transform")
    parser.add_argument('--augment', type=int, default=1, help='data argment to increase robustness')
    parser.add_argument('--name', type=str, default='train', help='name of the experiment')
    parser.add_argument('--note', type=str, default='', help='notation of the experiment')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--beta', default=1, type=float, help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')
    args = parser.parse_args()
    args.feature_transform, args.augment = bool(args.feature_transform), bool(args.augment)
    ### Set random seed
    args.seed = args.seed if args.seed > 0 else random.randint(1, 10000)

    # dataset path
    DATA_PATH = '/home/zjl/data/modelnet40_normal_resampled/'

    ########################################
    ## Intiate model
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 40

    if args.model == 'pointnet_rcutmix':
        model = PointNetCls(num_classes, args.feature_transform)
        model = model.to(device)
    elif args.model == 'pointnet2_rcutmix':
        model = PointNet2ClsMsg(num_classes)
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.model == 'dgcnn_rcutmix':
        model = DGCNN(num_classes)
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.model == 'rscnn_rcutmix':
        from models.rscnn import RSCNN  ## use torch 0.4.1.post2
        import models.rscnn_utils.pointnet2_utils as pointnet2_utils
        import models.rscnn_utils.pytorch_utils as pt_utils

        model = RSCNN(num_classes)
        model = model.to(device)
        model = nn.DataParallel(model)

    if len(args.resume) > 1:
        print('=====> Loading from checkpoint...')
        checkpoint = torch.load('checkpoints/%s.pth' % args.resume)
        args = checkpoint['args']

        torch.manual_seed(args.seed)
        print("Random Seed: ", args.seed)

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        acc_list = checkpoint['acc_list']
        print('Successfully resumed!')

    else:
        print('=====> Building new model...')
        torch.manual_seed(args.seed)
        print("Random Seed: ", args.seed)

        if args.model == 'dgcnn_rcutmix':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 100, momentum=0.9, weight_decay=1e-4)
            scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 250, eta_min=1e-3)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-4
            )
            scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        START_EPOCH = 0
        acc_list = [0]

        print('Successfully built!')

    ########################################
    ## Load data
    ########################################
    print('======> Loading data')

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='train',
                                        normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='test',
                                        normal_channel=args.normal)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                              num_workers=4, drop_last=False)

    PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()
    print('======> Successfully loaded!')

    gen_train_log(args)
    logname = ('logs_train/%s_%s_%s.csv' % (args.data, args.model, args.name))

    ########################################
    ## Train
    ########################################
    if args.model == 'dgcnn_rcutmix':
        criterion = cal_loss
    else:
        criterion = F.cross_entropy  # nn.CrossEntropyLoss()

    if args.resume == '/':
        log_row(logname, ['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'learning Rate'])


    for epoch in range(START_EPOCH, args.epochs):
        print('\nEpoch: %d' % epoch)
        scheduler_c.step(epoch)
        model.train()

        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            points, label = data
            points, label = points.to(device), label.to(device)[:, 0]
            points = PointcloudScaleAndTranslate(points)
            target = label

            if args.model == 'rscnn_rcutmix':
                fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(args.num_points, args.num_points, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                # cutmix
            optimizer.zero_grad()
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                B = points.size()[0]

                rand_index = torch.randperm(B).cuda()
                target_a = target
                target_b = target[rand_index]

                point_a = torch.zeros(B, 1024, 3)
                point_b = torch.zeros(B, 1024, 3)
                point_c = torch.zeros(B, 1024, 3)
                point_a = points
                point_b = points[rand_index]
                point_c = points[rand_index]
                point_a, point_b, point_c = point_a.to(device), point_b.to(device), point_c.to(device)

                remd = emd.emdModule()
                remd = remd.cuda()
                dis, ind = remd(point_a, point_b, 0.005, 300)
                for ass in range(B):
                    point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

                int_lam = int(args.num_points * lam)
                int_lam = max(1, int_lam)
                gamma = np.random.choice(args.num_points, int_lam, replace=False, p=None)
                for i2 in range(B):
                    points[i2, gamma, :] = point_c[i2, gamma, :]

                # adjust lambda to exactly match point ratio
                lam = int_lam * 1.0 / args.num_points
                points = points.transpose(2, 1)
                pred, trans_feat = model(points)
                loss = criterion(pred, target_a.long()) * (1. - lam) + criterion(pred, target_b.long()) * lam
            else:
                points = points.transpose(2, 1)
                pred, trans_feat = model(points)
                loss = criterion(pred, target.long())


            if args.feature_transform and args.model == 'pointnet_rcutmix':
                loss += feature_transform_regularizer(trans_feat) * args.lambda_ft
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct += pred_choice.eq(label.data).cpu().sum()
            total += label.size(0)
            progress_bar(i, len(train_loader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                         % (loss.item() / (i + 1), 100. * correct.item() / total, correct, total))

        train_loss, train_acc = loss.item() / (i + 1), 100. * correct.item() / total

        ### Test in batch

        test_loss, test_acc = test(model, test_loader, criterion)
        acc_list.append(test_acc)
        print('the best test acc is', max(acc_list))

        ### Keep tracing
        log_row(logname, [epoch, train_loss, train_acc, test_loss, test_acc,
                          optimizer.param_groups[0]['lr'], max(acc_list), np.argmax(acc_list) - 1])
        save_ckpt(args, epoch, model, optimizer, acc_list)



