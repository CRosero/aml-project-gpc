## TODO: put here all the training function and call them from outside

import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, colour_code_segmentation
from loss import DiceLoss
import torch.cuda.amp as amp
import os
import os.path as osp
import random
import matplotlib.pyplot as plt
import collections
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils import data
from PIL import Image
import json
# Validation function
from eval import val
# Dataset class:
from dataset.cityscapesDataSet import cityscapesDataSet



## -- TRAINING --

def train(args, model, epoch_start_i, optimizer, dataloader_train, dataloader_val, miou_init=0):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()

    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    max_miou = miou_init
    step = 0
    
    for epoch in range(epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.long().cuda()
  
            optimizer.zero_grad()
            
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            #tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
            
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            print("Saving checkpoint")
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.module.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()
                        },
                       os.path.join(args.save_model_path, 'latest_CE_loss.pth'))


        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save({
                      'miou': miou,
                      'epoch': epoch,
                      'model_state_dict': model.module.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()
                        },
                       os.path.join(args.save_model_path, 'best_CE_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument("--iter-size", type=int, default=1, help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='content/data', help='path of training data')
    #parser.add_argument("--data-list", type=str, default='/gdrive/MyDrive/data/Cityscapes/train.txt', help="Path to the file listing the images in the source dataset.")
    #parser.add_argument("--data-list-val", type=str, default='/gdrive/MyDrive/data/Cityscapes/val.txt', help="Path to the file listing the image in the test subset.")
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument("--num-steps", type=int, default=250000, help="Number of training steps.")
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--set-type", type=str, default='train', help="choose adaptation set.")
    parser.add_argument('--pretrained_best_model_path', type=str, default=None, help='path to the best pretrained model')
    parser.add_argument('--load_best_model', type=bool, default=False, help='load or not the best model from the saved checkpoint ')
    parser.add_argument('--load_last_model', type=bool, default=False, help='load or not the last model from the saved checkpoint ')


    args = parser.parse_args(params)

    # create dataset and dataloader
    data_root_path = os.path.join(args.data, args.dataset) # /content/data/Cityscapes
    train_path = os.path.join(data_root_path, "train.txt") # /content/data/Cityscapes/train.txt
    val_path = os.path.join(data_root_path, "val.txt")   # /content/data/Cityscapes/val.txt
    info_path = os.path.join(args.data, args.dataset, "info.json") # /content/data/Cityscapes/info.json 
    
    # preprocessing informations:
    input_size = (int(args.crop_width), int(args.crop_height))
    f = open(info_path)
    info = json.load(f)
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    img_mean = np.array(img_mean, dtype=np.float32)
    
    # create dataloaders
    train_dataset = cityscapesDataSet(root=data_root_path,
                                      list_path = train_path,
                                      info_json = info,
                                      crop_size=input_size,
                                      scale=args.random_scale, 
                                      mirror=args.random_mirror, 
                                      mean=img_mean)
   
    
    val_dataset = cityscapesDataSet(root=data_root_path,
                                    list_path = val_path,
                                    info_json = info,
                                    crop_size=input_size,
                                    scale=False,
                                    mirror=args.random_mirror, 
                                    mean=img_mean)
    
    print(f'train_dataset: {len(train_dataset)}')
    print(f'val_dataset: {len(val_dataset)}')
    image, label = train_dataset[0]
    print(f'images shape: {image.shape}')
    print(f'label shape: {label.shape}')
    
    # Define dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=args.num_workers, pin_memory=True)
    # (batch size for dataloader_val must be 1)
    dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    model = BiSeNet(args.num_classes, args.context_path)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    epoch_start_i = 0
    miou_init = 0

    # load pretrained model if exists
    if (args.load_last_model) and (args.pretrained_model_path is not None) and (os.path.isfile(args.pretrained_model_path)):
        print('load model from %s ...' % args.pretrained_model_path)
        checkpoint= torch.load(args.pretrained_model_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start_i = int(checkpoint['epoch'])+1
        print('Done! Now starting from epoch:', epoch_start_i)

        if not os.path.isfile(args.pretrained_best_model_path):
            miou_init=0
        else:
          miou_init = torch.load(args.pretrained_best_model_path)['miou']
          print('getting best miou from lasting runs: ', miou_init)

    # load best pretrained model if exists
    if (args.load_best_model)and (args.pretrained_best_model_path is not None) and (os.path.isfile(args.pretrained_best_model_path)):
        print('load best model from %s ...' % args.pretrained_best_model_path)
        checkpoint= torch.load(args.pretrained_best_model_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start_i = int(checkpoint['epoch'])
        print('Best results reached at epoch:', epoch_start_i)
        miou_init = torch.load(args.pretrained_best_model_path)['miou']
        print('getting best miou from last runs: ', miou_init)
        

    # train
    train(args, model, epoch_start_i, optimizer, dataloader_train, dataloader_val, miou_init)

    val(args, model, dataloader_val)

if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--data', "/content/data",
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',  
        '--save_model_path', '/gdrive/MyDrive/Project_AML/Models/checkpoints_18_sgd-epochs_50-batch_4-crop_1024_512',
        '--pretrained_model_path', '/gdrive/MyDrive/Project_AML/Models/checkpoints_101_sgd-batch_8-crop_1024_512-TESTINDEX/latest_CE_loss.pth',
        '--context_path', 'resnet18',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        '--checkpoint_step', '2',        
        '--pretrained_best_model_path', '/gdrive/MyDrive/Project_AML/Models/checkpoints_101_sgd-batch_8-crop_1024_512-TESTINDEX/best_CE_loss.pth',
        '--validation_step', '2',
        '--crop_width', '1024',
        '--crop_height', '512',
        '--load_best_model', False,
        '--load_last_model', False
            ]
    main(params)
