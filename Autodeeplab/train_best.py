import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


from tqdm import tqdm
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self,args):
        warnings.filterwarnings('ignore')
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)
        if args.dataset == 'pascal':
            raise NotImplementedError
        elif args.dataset == 'cityscapes':
            kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
            dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
            args.num_classes = num_classes
        elif args.dataset == 'marsh' :
            kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
            dataset_loader,val_loader, test_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
            args.num_classes = num_classes
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

        if args.backbone == 'autodeeplab':
            model = Retrain_Autodeeplab(args)
        else:
            raise ValueError('Unknown backbone: {}'.format(args.backbone))

        if args.criterion == 'Ohem':
            args.thresh = 0.7
            args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
            args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
        criterion = build_criterion(args)

        model = nn.DataParallel(model).cuda()
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

        max_iteration = len(dataset_loader) * args.epochs
        scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))

        start_epoch = 0

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
                self.best_pred = checkpoint['best_pred']
            else:
                raise ValueError('=> no checkpoint found at {0}'.format(args.resume))
        ##mergee 
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        #kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = dataset_loader,val_loader, test_loader, num_classes

        self.criterion = criterion
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        #self.scheduler = scheduler
        self.scheduler = LR_Scheduler("poly",args.lr, args.epochs, len(self.train_loader)) #removed None from second parameter. 

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            cur_iter = epoch * len(self.train_loader) + i
            #self.scheduler(self.optimizer, cur_iter)# this scheduler did not work. let try other one for say 500 epochs.
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                #print("I was here!!")
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        
        # save checkpoint every epoch
        is_best = False
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU,IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Classwise_IoU:")
        print(IoU)
        print('Loss: %.3f' % test_loss)
        print(self.evaluator.confusion_matrix)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    args = obtain_retrain_autodeeplab_args()
    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'marsh' : 0.1,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * args.batch_size)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.writer.close()

if __name__ == "__main__":
    main()
