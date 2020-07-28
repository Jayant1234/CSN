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
            model.load_state_dict(torch.load(r"./run/marsh/deeplab-autodeeplab/model_best.pth.tar")['state_dict'], strict=False)
        else:
            raise ValueError('Unknown backbone: {}'.format(args.backbone))

       optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)


        if args.criterion == 'Ohem':
            args.thresh = 0.7
            args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
            args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
        criterion = build_criterion(args)
		
		
        model = nn.DataParallel(model).cuda()
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

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
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
        mIoU = self.evaluator.Mean_Intersection_over_Union()
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

        del confusion[1]
        for i in range(len(confusion)):
            del confusion[i][1]		
 		
        cm = np.array(confusion)
        cm.astype(int)
		
        print(cm)
        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) - true_pos
        false_neg = np.sum(cm, axis=1) - true_pos

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        print('Precision:',precision)
        print('Recall:',recall)

        precision_one = np.sum(true_pos) / (np.sum(true_pos) + np.sum(false_pos))
        recall_one = np.sum(true_pos )/ (np.sum(true_pos) + np.sum(false_neg))
        print('Precision:',false_pos)
        print('Recall:',false_neg)


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
    trainer.validation(epoch)

if __name__ == "__main__":
    main()
