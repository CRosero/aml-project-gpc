from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.cuda.amp as amp

import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils import data
import torch.nn.functional as F


from utils.utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, colour_code_segmentation,poly_lr_scheduler
from utils.loss import CrossEntropy2d,DiceLoss
import numpy as np
import os
import os.path as osp
import random
import matplotlib.pyplot as plt
import collections
from PIL import Image
from torchinfo import summary
import json
import argparse
from tqdm import tqdm

# Dataset class:
from dataset.cityscapesDataSet import cityscapesDataSet
from dataset.GTA5DataSet import GTA5DataSet
# Discriminator
from model.discriminator import FCDiscriminator, LightWeightFCDiscriminator
# Network
from model.build_BiSeNet import BiSeNet
# Validation function
from eval import val
# FDA
from utils.FDA import FDA_source_to_target

data_path = "/content/data"

def enable_cuda(obj, gpu):
  if torch.cuda.is_available():
    return obj.cuda(gpu)
  else:
    return obj
    
def loss_calc(pred, labels, gpu, ignore_label=255):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    labels = Variable(labels.long()).cuda(gpu)
    labels = enable_cuda(labels, gpu)
    criterion = CrossEntropy2d(ignore_label= ignore_label)
    criterion = enable_cuda(criterion, gpu)

    return criterion(pred, labels)

def train_DA(args, model, model_D, optimizer,optimizer_D, sourceloader, targetloader, targetloaderVal, miou_init=0, iter_start_i=0):
  # paths
  model_description = "DA_checkpoints-" + "Light_Discriminator-" + "FDA_Blur-Beta_0.05"
  save_models_path = args.save_models_path + model_description  
  save_tb_path = args.save_tb_path + model_description 
  
  # labels for adversarial training
  source_label_id = 0
  target_label_id = 1

  scaler = amp.GradScaler()

  if args.gan == 'Vanilla':
    bce_loss = torch.nn.BCEWithLogitsLoss()
  elif args.gan == 'LS':
    bce_loss = torch.nn.MSELoss()

  writer = SummaryWriter(log_dir = save_tb_path)
  max_miou = miou_init
  
  if args.FDA:
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
    mean_img = torch.zeros(1, 1)
  
  sourceloader_iter = enumerate(sourceloader)
  targetloader_iter = enumerate(targetloader)
  
  for i_iter in range(iter_start_i, args.num_steps):

    model.train() 
    model_D.train()   

    # initialize loss values to 0
    loss_seg_value = 0
    loss_adv_target_value = 0
    loss_D_value = 0

    poly_lr_scheduler(optimizer, args.learning_rate, iter=i_iter, max_iter=args.num_steps)
    poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=i_iter, max_iter=args.num_steps)
    
    tq = tqdm(total=args.iter_size*args.batch_size)
    tq.set_description('iter %d / %d' % (i_iter, args.num_steps))
    
    for sub_i in range(args.iter_size): 
       
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        # train Generator

        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        
        # get batch from dataloaders
        try:
          _, batch_source = next(sourceloader_iter)  # new batch source
        except:
          sourceloader_iter = enumerate(sourceloader)
          _, batch_source = next(sourceloader_iter)
        
        try:
          _, batch_target = next(targetloader_iter) # new batch target
        except:
          targetloader_iter = enumerate(targetloader)
          _, batch_target = next(targetloader_iter) # new batch target

        source_images, source_labels = batch_source
        target_images, _ = batch_target

        if args.FDA: 
          #----------------------------- FDA ---------------------------------#
          if mean_img.shape[-1] < 2:
              B, C, H, W = source_images.shape
              mean_img = IMG_MEAN.repeat(B,1,H,W)
          #-------------------------------------------------------------------#
          # 1. source to target, target to target
          src_in_trg = FDA_source_to_target( source_images, target_images, L=args.LB )    # src_lbl
          trg_in_trg = target_images
          # 2. subtract mean
          source_images = src_in_trg.clone() - mean_img   # src, src_lbl
          target_images = trg_in_trg.clone() - mean_img   # trg, trg_lbl

          #-------------------------------------------------------------------#
        
        # train with source images and labels
        source_images = Variable(source_images)
        source_images = enable_cuda(source_images, args.gpu)
        source_labels = Variable(source_labels)
        source_labels = enable_cuda(source_labels, args.gpu)

        with amp.autocast():
          pred_source_result, pred_source_1, pred_source_2 = model(source_images)
          loss1 = loss_calc(pred_source_result, source_labels, args.gpu, args.ignore_label)
          loss2 = loss_calc(pred_source_1, source_labels, args.gpu, args.ignore_label)
          loss3 = loss_calc(pred_source_2, source_labels, args.gpu, args.ignore_label)
          loss_seg = loss1 + loss2 + loss3

        # proper normalization
        loss_seg = loss_seg #/ args.iter_size
        scaler.scale(loss_seg).backward()

        loss_seg_value += loss_seg.data.cpu().numpy() 

        # train with target images
        target_images = Variable(target_images).cuda(args.gpu)

        pred_target_result, pred_target_1, pred_target_2 = model(target_images)

        # generator vs. discriminator 
        with amp.autocast():
          D_out = model_D(F.softmax(pred_target_result))

          loss_adv_target = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label_id)).cuda(args.gpu))
          loss = args.lambda_adv_target * loss_adv_target
          loss = loss #/ args.iter_size
        
        scaler.scale(loss).backward()
        
        loss_adv_target_value += loss_adv_target.data.cpu().numpy() #/ args.iter_size

        # train discriminator

        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        pred_source_result = pred_source_result.detach()
        with amp.autocast():
          D_out_source = model_D(F.softmax(pred_source_result))
       
          loss_D_source = bce_loss(D_out_source, Variable(torch.FloatTensor(D_out_source.data.size()).fill_(source_label_id)).cuda(args.gpu))
          loss_D_source = loss_D_source / 2 #/ args.iter_size
        
        scaler.scale(loss_D_source).backward()

        loss_D_value += loss_D_source.data.cpu().numpy()

        # train with target
        pred_target_result = pred_target_result.detach()

        with amp.autocast():
          D_out_target = model_D(F.softmax(pred_target_result))

          loss_D_target = bce_loss(D_out_target, Variable(torch.FloatTensor(D_out_target.data.size()).fill_(target_label_id)).cuda(args.gpu))
          loss_D_target = loss_D_target  / 2 #/ args.iter_size
        
        scaler.scale(loss_D_target).backward()

        loss_D_value += loss_D_target.data.cpu().numpy()
        
        scaler.step(optimizer)
        scaler.step(optimizer_D) 
        scaler.update()

        tq.update(args.batch_size) 
      

    tq.close()
    print("")
    print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv = {3:.3f}, loss_D = {4:.3f}'.format(i_iter, args.num_steps, loss_seg_value/args.iter_size, loss_adv_target_value/args.iter_size, loss_D_value/args.iter_size))
    
    writer.add_scalar('loss_seg_value', loss_seg_value/args.iter_size, i_iter)
    writer.add_scalar('loss_adv_target_value', loss_adv_target_value/args.iter_size, i_iter)
    writer.add_scalar('loss_D_value', loss_D_value/args.iter_size, i_iter)


    if i_iter % args.save_pred_every == 0 and i_iter != 0:
        #print(" Saving checkpoint in ", save_models_path, "latest_CE_loss.pth")
        if not os.path.isdir(save_models_path):
            os.mkdir(save_models_path)
        torch.save({
                  'iter': i_iter,
                  'segNet_state_dict': model.state_dict(),
                  'D_state_dict': model_D.state_dict(),
                  'optimizer_seg_state_dict': optimizer.state_dict(),
                  'optimizer_D_state_dict': optimizer_D.state_dict(),
                  'max_miou' : max_miou,
                    },
                    os.path.join(save_models_path, 'latest_CE_loss.pth'))
    
    if i_iter % args.validation_step == 0 and i_iter != 0:
        #print(" doing validation at iter ", i_iter)
        precision, miou = val(args, model, targetloaderVal)
        if miou > max_miou:
            max_miou = miou
            #print(" Saving checkpoint in ", save_models_path, "best_CE_loss.pth")
            os.makedirs(save_models_path, exist_ok=True)
            torch.save({
                  'iter': i_iter,
                  'segNet_state_dict': model.state_dict(),
                  'D_state_dict': model_D.state_dict(),
                  'optimizer_seg_state_dict': optimizer.state_dict(),
                  'optimizer_D_state_dict': optimizer_D.state_dict(),
                  'max_miou' : max_miou,
                    },
                    os.path.join(save_models_path, 'best_CE_loss.pth'))
        writer.add_scalar('precision', precision, i_iter)
        writer.add_scalar('miou', miou, i_iter)

  return


def get_arguments(params=[]):
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
        
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='BiseNet',
                        help="available options : BiseNet")
    parser.add_argument("--target", type=str, default='Cityscapes',
                        help="available options : Cityscapes")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default='',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore-label", type=int, default= 255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default='1024,512',
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--input-size-target", type=str, default='1024,512',
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-2,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-target", type=float, default=0.001,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of training steps.")
    parser.add_argument("--iter-size", type=int, default=125,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-steps-stop", type=int, default=150,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-pred-every", type=int, default=10,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--gan", type=str, default='Vanilla',
                        help="choose the GAN objective.")
    parser.add_argument('--context_path', type=str, default='resnet18',
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument("--light_discriminator", type=bool, default=False, 
                        help="using discriminator with lightweight depthwise-separable convolutions")
    parser.add_argument('--load_pretrained_models', type=bool, default=False, help='load or not the pretrained models from the saved checkpoint ')
    parser.add_argument('--pretrained_models_path', type=str, default="", help='path to pretrained models')
    parser.add_argument('--save_models_path', type=str, default="", help='path to save models')
    parser.add_argument('--save_tb_path', type=str, default=None, help='path to save tensorboard graphs')

    
    parser.add_argument('--FDA', type=bool, default=True, help='whether to use FDA to transform source images')
    parser.add_argument("--LB", type=float, default=0.1, help="beta for FDA")



    args = parser.parse_args(params)
    return args

def main(params):
  """Create the model and start the training."""
  args = get_arguments(params)

  # Set random seed
  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed(args.random_seed)
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)

  # input sizes
  w, h = map(int, args.input_size.split(','))
  input_size = (w, h)

  w, h = map(int, args.input_size_target.split(','))
  input_size_target = (w, h)

  cudnn.benchmark = True
  cudnn.enabled = True
  gpu = args.gpu

  # Create network
  if args.model == 'BiseNet':
    model = BiSeNet(num_classes=args.num_classes, context_path= args.context_path)

  # init D
  if args.light_discriminator == False:
    print("Using a fully convolutional discriminator")
    model_D = FCDiscriminator(num_classes=args.num_classes)
  else:
    print("Using a discriminator with lightweight depthwise-separable convolution")
    model_D = LightWeightFCDiscriminator(num_classes=args.num_classes)

  model = enable_cuda(model, args.gpu)
  model_D = enable_cuda(model_D, args.gpu)

  # Printing statistics
  '''
  print("Segmentation Network\n")
  print(summary(enable_cuda(model.eval(), args.gpu), input_size=(args.batch_size, 3, input_size[0], input_size[1])))

  if args.light_discriminator == False: 
    print("Adversarial discriminator Architecture\n")
  else:
    print("Lightweight Adversarial Domain Adaptation\n")
  print(summary(enable_cuda(model_D.eval(), args.gpu), input_size=(args.batch_size, 19, input_size[0], input_size[1])))
  '''
  

  # Path
  source_data_root_path = os.path.join(args.data_dir, "GTA5") # /content/data/GTA5
  target_data_root_path = os.path.join(args.data_dir, args.target) # /content/data/Cityscapes
  source_train_path = os.path.join(source_data_root_path, "train.txt") # /content/data/GTA5/train.txt
  target_root_path = os.path.join(target_data_root_path,  "train.txt")   # /content/data/Cityscapes/train.txt
  val_root_path = os.path.join(target_data_root_path,  "val.txt")   # /content/data/Cityscapes/train.txt
  info_path = os.path.join(source_data_root_path,  "info.json") # /content/data/GTA/info.json 

  info_json = json.load(open(info_path))

  # Image mean
  IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
  # Zero mean
  IMG_MEAN_ZERO = np.array((0.0, 0.0, 0.0), dtype=np.float32)
  IMG_MEAN_VAL = IMG_MEAN

  if (args.FDA):
    img_mean = IMG_MEAN_ZERO # From FDA original code: "use the original images for FDA, then do mean subtraction, normalization, etc. Otherwise, will be numerical artifact"
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
  else:
    img_mean = IMG_MEAN

  # Augmentation
  #AUG_VALUES = None
  AUG_VALUES = {
        "hor_flip": True,
        "blur": True,
        "prob" : 0.5, 
        "kernel_size" : 9,
        "sigma" : (1,2)
    }
  # Datasets  
  source_dataset = GTA5DataSet(source_data_root_path, source_train_path, info_json, crop_size=input_size, mean=img_mean, augmentation=AUG_VALUES) #, max_iters=args.num_steps * args.iter_size * args.batch_size)
  target_dataset = cityscapesDataSet(target_data_root_path, target_root_path, info_json, crop_size=input_size_target, mean=img_mean, augmentation=AUG_VALUES ) #, max_iters=args.num_steps * args.iter_size * args.batch_size)
  target_dataset_Val = cityscapesDataSet(target_data_root_path, val_root_path, info_json, crop_size=input_size_target, mean=IMG_MEAN_VAL)

  print("GTA: ", len(source_dataset))
  print("Cityscapes: ", len(source_dataset))
  img,label = source_dataset[0]
  print ("GTA image", img.shape )
  print ("GTA label", label.shape )
  img, _ = target_dataset[0]
  print ("Cityscapes image", img.shape )

  
  # Create DataLoaders
  sourceloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
  targetloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
  targetloaderVal = data.DataLoader(target_dataset_Val, batch_size=1, num_workers=args.num_workers, pin_memory=True)

  # Optimizer

  optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  optimizer.zero_grad()

  optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
  optimizer_D.zero_grad()

  # to keep track of best miou
  max_miou = 0
  # start iteration from:
  iter_start_i = 0

  # load pretrained model if exists
  checkpoint = None
  if (args.load_pretrained_models) and (args.pretrained_models_path is not None) and (os.path.isfile(args.pretrained_models_path)):
      print('load models from %s ...' % args.pretrained_models_path)
      checkpoint= torch.load(args.pretrained_models_path)
      model.load_state_dict(checkpoint['segNet_state_dict'])
      model_D.load_state_dict(checkpoint['D_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_seg_state_dict'])
      optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
      iter_start_i = int(checkpoint['iter'])+1
      max_miou = float(checkpoint['max_miou']) 
      print('Done! Loaded model trained until iter:', iter_start_i, "best miou so far:", max_miou)

  # train
  train_DA(args, model, model_D, optimizer,optimizer_D, sourceloader, targetloader, targetloaderVal, miou_init=max_miou, iter_start_i=iter_start_i)
  # final val
  val(args, model, targetloaderVal, save=True, save_path=data_path)


if __name__ == '__main__':
    params = [
              '--model', 'BiseNet',
              '--target', 'Cityscapes',
              '--batch-size', '4',
              '--num-workers', '8',
              '--data-dir', data_path,
              '--ignore-label', '255',
              '--input-size', '1024,512',
              '--input-size-target',  '1024,512',
              '--learning-rate', '2.5e-2',
              '--learning-rate-D', '1e-4',
              '--lambda-adv-target', '0.001',
              '--momentum', '0.9',
              '--power', '0.9',
              '--weight-decay','1e-4',
              '--num-classes', '19',
              '--num-steps', '51', # number of training step (over a iter_size batches)
              '--gpu', '0',
              '--gan', 'Vanilla',
              '--context_path', 'resnet18', # or 'resnet101'
              '--save-pred-every', '2',
              '--validation_step', '2',
              '--light_discriminator', 'True',
              '--load_pretrained_models','True',
              '--pretrained_models_path', '/gdrive/MyDrive/Project_AML/Models/adversarialDA/DA_checkpoints-Light_Discriminator-FDA_Blur-Beta_0.05/latest_CE_loss.pth',
              '--save_models_path', '/gdrive/MyDrive/Project_AML/Models/adversarialDA/',
              '--save_tb_path', '/gdrive/MyDrive/Project_AML/Graphs/adversarialDA/',
              
              '--FDA', 'True',
              '--LB', '0.05',

              '--iter-size', '125'



    ]
    main(params)