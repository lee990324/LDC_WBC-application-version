
from __future__ import print_function

import argparse
import os
import time, platform

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile

from dataset import DATASET_NAMES, TrainDataset, TestDataset, dataset_info, MODEL_DIR
from loss import *
from model import LDC
from utils.img_processing import (image_normalization, save_image_batch_to_disk,
                   visualize_result, count_parameters)

def train_one_epoch(epoch, dataloader, model, criterions, optimizer, device,
                    log_interval_vis, args=None):

    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder,exist_ok=True)

    if isinstance(criterions, list):
        criterion1, criterion2 = criterions
    else:
        criterion1 = criterions

    # Put model in training mode
    model.train()

    l_weight0 = [0.7,0.7,1.1,0.7,1.3] # for bdcn loss2-B4
    # l_weight0 = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3] # for bdcn loss2-B6

    l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
                [0.1, 1.], [0.1, 1.], [0.1, 1.],
                [0.01, 4.]]  # for cats loss
    loss_avg =[]
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        preds_list = model(images)

        # loss = sum([criterion2(preds, labels,l_w) for preds, l_w in zip(preds_list[:-1],l_weight0)]) # bdcn_loss2
        loss = sum([criterion1(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())
        if epoch==0 and (batch_id==100):
            tmp_loss = np.array(loss_avg).mean()

        if batch_id % 10 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), format(loss.item(),'.4f')))
        if batch_id % log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[2])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), loss.item())

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None):
    # XXX This is not really validation, but testing

    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images)
            # print('pred shape', preds[0].shape)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names,img_shape=image_shape,
                                     arg=arg)


def test(checkpoint_path, dataloader, model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "TEST":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            print(f"{file_names}: {images.shape}")
            # if batch_id==0:
            #     mac,param = profile(model,inputs=(images,))
            #     end = time.perf_counter()
            #     if device.type == 'cuda':
            #         torch.cuda.synchronize()
            #     preds = model(images)
            #     if device.type == 'cuda':
            #         torch.cuda.synchronize()
            # else:
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()
    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))
    # print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def testPich(checkpoint_path, dataloader, model, device, output_dir, args):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "TEST":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            images2 = images[:, [1, 0, 2], :, :]  #GBR
            # images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images)
            preds2 = model(images2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def parse_args():
    parser = argparse.ArgumentParser(description='LDC trainer.')
    parser.add_argument('--train_dataset',
                        type=int,
                        default=0, # BIPED
                        help='Choose a dataset for testing: 0 - 8')

    TRAIN_DATA = DATASET_NAMES[parser.parse_args().train_dataset]
    train_inf = dataset_info(TRAIN_DATA)
    train_dir = train_inf['data_dir']
    
    parser.add_argument('--val_dataset',
                        type=int,
                        default=1, # CUSTOM
                        help='Choose a dataset for testing: 0 - 8')
    
    VAL_DATA = DATASET_NAMES[parser.parse_args().val_dataset]
    val_inf = dataset_info(VAL_DATA)
    val_dir = val_inf['data_dir']

    parser.add_argument('--test_dataset',
                        type=int,
                        default=2, # CUSTOM
                        help='Choose a dataset for testing: 0 - 8')
    
    TEST_DATA = DATASET_NAMES[parser.parse_args().test_dataset]
    test_inf = dataset_info(TEST_DATA)
    test_dir = test_inf['data_dir']

    is_testing =False
    model_dir = MODEL_DIR
    parser.add_argument('--is_testing',type=bool,
                        default=is_testing,
                        help='Script in testing mode.')
    parser.add_argument('--model_dir',
                        type=str,
                        default=model_dir,
                        help='model_dir.')
    

    # Input DIR
    parser.add_argument('--train_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the train data.')
    parser.add_argument('--val_dir',
                        type=str,
                        default=val_inf['data_dir'],
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--test_dir',
                        type=str,
                        default=test_inf['data_dir'],
                        help='the path to the directory with the input data for test.')
    # Output DIR
    parser.add_argument('--output_dir',
                        type=str,
                        default='result_output',
                        help='the path to output the results.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    # DataSet
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--val_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=VAL_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    # Data size
    parser.add_argument('--img_width',
                        type=int,
                        default=352,
                        help='Image width for training.') # BIPED 352 BSDS 352/320 MDBD 480
    parser.add_argument('--img_height',
                        type=int,
                        default=352,
                        help='Image height for training.') # BIPED 480 BSDS 352/320
    parser.add_argument('--val_img_width',
                        type=int,
                        default=val_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--val_img_height',
                        type=int,
                        default=val_inf['img_height'],
                        help='Image height for testing.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf['img_height'],
                        help='Image height for testing.')
    #Option
    parser.add_argument('--predict_all',
                        type=bool,
                        default=False,
                        help='True: Generate all LDC outputs in all_edges ')
    parser.add_argument('--double_img',
                        type=bool,
                        default=False,
                        help='True: use same 2 imgs changing channels')  # Just for test
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=100,
                        help='The NO B to wait before printing test predictions. 200')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        metavar='N',
                        help='Number of training epochs (default: 25).')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Initial learning rate. =5e-5')
    parser.add_argument('--lrs', default=[25e-4,5e-4,1e-5], type=float,
                        help='LR for set epochs')
    parser.add_argument('--wd', type=float, default=0., metavar='WD',
                        help='weight decay (Good 5e-6)')
    parser.add_argument('--adjust_lr', default=[6,12,18], type=int,
                        help='Learning rate step size.')  # [6,9,19]
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68,137.86],
                        type=float)  # [103.939,116.779,123.68,137.86] [104.00699, 116.66877, 122.67892]
    # BRIND mean = [104.007, 116.669, 122.679, 137.86]
    # BIPED mean_bgr processed [160.913,160.275,162.239,137.86]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = LDC().to(device)
    ini_epoch =0

    # Testing
    if args.is_testing:
        dataset_test = TestDataset(args.test_dir,
                                test_data=args.test_data,
                                img_width=args.test_img_width,
                                img_height=args.test_img_height,
                                mean_bgr=args.mean_pixel_values[0:3] if len(
                                    args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                arg=args
                                )
        dataloader_test = DataLoader(dataset_test,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.workers)
        output_dir = os.path.join(args.res_dir, args.train_data+"2"+ args.test_data)
        print(f"output_dir: {output_dir}")
        if args.double_img:
            # run twice the same image changing the image's channels
            testPich(args.checkpoint_path, dataloader_test, model, device, output_dir, args)
        else:
            test(args.checkpoint_path, dataloader_test, model, device, output_dir, args)

        # Count parameters:
        num_param = count_parameters(model)
        print('-------------------------------------------------------')
        print('LDC parameters:')
        print(num_param)
        print('-------------------------------------------------------')
        return
    
    # Training
    dataset_train = TrainDataset(args.train_dir,
                                    img_width=args.img_width,
                                    img_height=args.img_height,
                                    mean_bgr=args.mean_pixel_values[0:3] if len(
                                        args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                    arg=args
                                    )
    dataloader_train = DataLoader(dataset_train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers)
    dataset_val = TestDataset(args.val_dir,
                            test_data=args.val_data,
                            img_width=args.val_img_width,
                            img_height=args.val_img_height,
                            mean_bgr=args.mean_pixel_values[0:3] if len(
                                args.mean_pixel_values) == 4 else args.mean_pixel_values,
                            arg=args
                            )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)

    criterion1 = cats_loss #bdcn_loss2
    criterion2 = bdcn_loss2#cats_loss#f1_accuracy2
    criterion = [criterion1,criterion2]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)

    # Count parameters:
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('LDC parameters:')
    print(num_param)
    print('-------------------------------------------------------')

    # Main training loop
    seed=1021
    adjust_lr = args.adjust_lr
    k=0
    set_lr = args.lrs#[25e-4, 5e-6]
    for epoch in range(ini_epoch,args.epochs):
        if epoch%7==0:

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # adjust learning rate
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = set_lr[k]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2
                k+=1
        # Create output directories

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.val_data + '_res')
        os.makedirs(output_dir_epoch,exist_ok=True)
        os.makedirs(img_test_dir,exist_ok=True)

        avg_loss =train_one_epoch(epoch,dataloader_train,
                        model, criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        print('Last learning rate> ', optimizer.param_groups[0]['lr'])

    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('LDC parameters:')
    print(num_param)
    print('-------------------------------------------------------')

if __name__ == '__main__':
    args = parse_args()
    main(args)