import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils.utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou, colour_code_segmentation
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils import data
import torch.nn.functional as F

# Dataset class:
from dataset.cityscapesDataSet import cityscapesDataSet

# Network
from model.build_BiSeNet import BiSeNet


## -- VALIDATION --
def val(args, model, dataloader, save=False, batch_size=1):

    #TODO: prendere dal json
    palette = [[128,64,128],[244,35,232], [70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32],[0,0,0]]
    num = list(range(0, len(palette)-1))
    num.append(255)
    dictionary = dict(zip(num, palette)) 
    

    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        tq =tqdm(total=len(dataloader) * batch_size)
        tq.set_description('val')
        
        for i, (data, label) in enumerate(dataloader):
            tq.update(batch_size)
            label = label.type(torch.LongTensor)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)

            if save and i < 20:
              # save some images
              predict = colour_code_segmentation(np.array(predict), dictionary)
              label = colour_code_segmentation(np.array(label), dictionary)
              if not os.path.isdir("/content/cloned-repo/image_output"):
                os.mkdir("/content/cloned-repo/image_output")

              if not os.path.isdir("/content/cloned-repo/image_output/predict"):
                os.mkdir("/content/cloned-repo/image_output/predict")

              if not os.path.isdir("/content/cloned-repo/image_output/label"):
                os.mkdir("/content/cloned-repo/image_output/label")

              predictImage = Image.fromarray(predict.astype('uint8'), "RGB")
              predictImage.save("/content/cloned-repo/image_output/predict/" + str(i) + ".png")

              labelImage = Image.fromarray(label.astype('uint8'), "RGB")
              labelImage.save("/content/cloned-repo/image_output/label/" + str(i) + ".png")
            
        
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist) #[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print("")
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def test(args,model,dataloader, info_json, save_path="", save=False, batch_size=1):
    #TODO: prendere dal json
    #palette = [[128,64,128],[244,35,232], [70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32],[0,0,0]]
    palette = info_json['palette']
    num = list(range(0, len(palette)-1))
    num.append(255)
    dictionary = dict(zip(num, palette)) 
    print('start test!')
    
    if (save):
      folder_predict =os.path.join(save_path, "predict")
      folder_labels =os.path.join(save_path, "labels")

      if not os.path.isdir(folder_predict):
        os.mkdir(folder_predict)

      if not os.path.isdir(folder_labels):
        os.mkdir(folder_labels)

    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm(total=len(dataloader) * batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))

        for i, (data, label) in enumerate(dataloader):
            tq.update(batch_size)
            label = label.type(torch.LongTensor)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)

            if save and i < 20:
              # save some images
              predict = colour_code_segmentation(np.array(predict), dictionary)
              label = colour_code_segmentation(np.array(label), dictionary)

              predictImage = Image.fromarray(predict.astype('uint8'), "RGB")
              predictImage.save(os.path.join(folder_predict, str(i) + ".png"))

              labelImage = Image.fromarray(label.astype('uint8'), "RGB")
              labelImage.save(os.path.join(folder_labels, str(i) + ".png"))
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist) #[:-1]
              
        miou_dict, miou = cal_miou(miou_list, info_json)
        print('')
        print('IoU for each class:')
        for key in miou_dict:
            print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('---------------------------')
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision

def get_arguments(params=[]):
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """   
    
    # basic parameters
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')   

    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')    
    parser.add_argument('--data', type=str, default='content/data', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')   
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed to have reproducible results.")
    parser.add_argument('--save_img_path', type=str, default=None, help='path to folder where to save imgs')   

    
    args = parser.parse_args(params)
    return args

def main(params):

    args = get_arguments(params)
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # create dataset and dataloader
    data_root_path = os.path.join(args.data, args.dataset) # /content/data/Cityscapes
    val_path = os.path.join(data_root_path, "val.txt")   # /content/data/Cityscapes/val.txt
    info_path = os.path.join(args.data, args.dataset, "info.json") # /content/data/Cityscapes/info.json 
    
    # preprocessing informations:
    input_size = (int(args.crop_width), int(args.crop_height))
    f = open(info_path)
    info = json.load(f)
    img_mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    img_mean = np.array(img_mean, dtype=np.float32)
    

    
    test_dataset = cityscapesDataSet(root=data_root_path,
                                    list_path = val_path,
                                    info_json = info,
                                    crop_size=input_size, 
                                    mean=img_mean)

    print(f'test_dataset: {len(test_dataset)}')
    image, label = test_dataset[0]
    print(f'images shape: {image.shape}')
    print(f'label shape: {label.shape}')
    
    # Define dataloaders

    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    model = BiSeNet(args.num_classes, args.context_path)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()


    # load pretrained model if exists
    
    print('load model from %s ...' % args.pretrained_model_path)
    checkpoint= torch.load(args.pretrained_model_path)
    try:
      model.module.load_state_dict(checkpoint['segNet_state_dict'])
      epoch_start_i = int(checkpoint['iter'])
    except:
      model.module.load_state_dict(checkpoint['model_state_dict'])
      epoch_start_i = int(checkpoint['epoch'])

    miou_init = float(checkpoint['max_miou'])
    print('Done!')
    print('Trained until Epoch:', epoch_start_i)
    print('- Best miou:', miou_init)

    if (args.save_img_path is not None):
      if not os.path.isdir(args.save_img_path):
        os.mkdir(args.save_img_path)
      # test
      test(args,model, dataloader_test, info, save=True, batch_size=1, save_path=args.save_img_path)
    else:
      # test
      test(args,model, dataloader_test, info, save=False, batch_size=1)



if __name__ == '__main__':
    params = [
        '--pretrained_model_path', '/gdrive/MyDrive/Project_AML/Models/segmentation/checkpoints-m_resnet18-sgd-e_50-b_4-c_1024_512/latest_CE_loss.pth',
        '--data', '/content/data',
        '--cuda', '0',
        '--context_path', 'resnet18',
        '--num_classes', '19',
        '--save_img_path', '/gdrive/MyDrive/Project_AML/Output/segmentation/noaug'
    ]

    main(params) 
