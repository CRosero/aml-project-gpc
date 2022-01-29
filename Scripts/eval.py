import torch
import argparse
import os
from torch.utils.data import DataLoader
from model.build_BiSeNet import BiSeNet
import numpy as np
from utils.utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, cal_miou, colour_code_segmentation
from tqdm import tqdm
from PIL import Image


## -- VALIDATION --
def val(args, model, dataloader, save=False):

    #TODO: prendere dal json
    palette = [[128,64,128],[244,35,232], [70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32],[0,0,0]]
    num = list(range(0, len(palette)-1))
    num.append(255)
    dictionary = dict(zip(num, palette)) 
    

    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        tq =tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('val')
        
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
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
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print()
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def test(model,dataloader, args, csv_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict)
            # predict = colour_code_segmentation(np.array(predict), label_info)

            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label)
            # label = colour_code_segmentation(np.array(label), label_info)

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou_dict, miou = cal_miou(miou_list, csv_path)
        print('IoU for each class:')
        for key in miou_dict:
            print('{}:{},'.format(key, miou_dict[key]))
        tq.close()
        print('precision for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='/path/to/data', help='Path of training data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    args = parser.parse_args(params)

    # create dataset and dataloader here
   
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # get label info
    # label_info = get_label_info(csv_path)
    # test
    test(model, dataloader, args, csv_path)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', 'path/to/ckpt',
        '--data', '/path/to/data',
        '--cuda', '0',
        '--context_path', 'resnet18',
        '--num_classes', '12'
    ]
    main(params)
