import sys
sys.path.append('/home/josmar/proyectos/yolact')

from data import *
from utils.augmentations import SSDAugmentation, BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from layers.output_utils import postprocess, undo_image_transformation #To vuisualize
from yolact import Yolact
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime
import wandb

# Oof
import eval as eval_script
import matplotlib.pyplot as plt

# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")


# parser = argparse.ArgumentParser(
#     description='Yolact Training Script')
# parser.add_argument('--batch_size', default=8, type=int,
#                     help='Batch size for training')
# parser.add_argument('--resume', default=None, type=str,
#                     help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
#                          ', the model will resume training from the interrupt file.')
# parser.add_argument('--start_iter', default=-1, type=int,
#                     help='Resume training at this iter. If this is -1, the iteration will be'\
#                          'determined from the file name.')
# parser.add_argument('--num_workers', default=4, type=int,
#                     help='Number of workers used in dataloading')
# parser.add_argument('--cuda', default=True, type=str2bool,
#                     help='Use CUDA to train model')
# parser.add_argument('--lr', '--learning_rate', default=None, type=float,
#                     help='Initial learning rate. Leave as None to read this from the config.')
# parser.add_argument('--momentum', default=None, type=float,
#                     help='Momentum for SGD. Leave as None to read this from the config.')
# parser.add_argument('--decay', '--weight_decay', default=None, type=float,
#                     help='Weight decay for SGD. Leave as None to read this from the config.')
# parser.add_argument('--gamma', default=None, type=float,
#                     help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
# parser.add_argument('--save_folder', default='weights/',
#                     help='Directory for saving checkpoint models.')
# parser.add_argument('--log_folder', default='logs/',
#                     help='Directory for saving logs.')
# parser.add_argument('--config', default=None,
#                     help='The config object to use.')
# parser.add_argument('--save_interval', default=10000, type=int,
#                     help='The number of iterations between saving the model.')
# parser.add_argument('--validation_size', default=5000, type=int,
#                     help='The number of images to use for validation.')
# parser.add_argument('--validation_iter', default=10000, type=int,
#                     help='Output validation information every n iterations. If -1, do no validation.')
# parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
#                     help='Only keep the latest checkpoint instead of each one.')
# parser.add_argument('--keep_latest_interval', default=100000, type=int,
#                     help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
# parser.add_argument('--dataset', default=None, type=str,
#                     help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
# parser.add_argument('--no_log', dest='log', action='store_false',
#                     help='Don\'t log per iteration information into log_folder.')
# parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
#                     help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
# parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
#                     help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
# parser.add_argument('--batch_alloc', default=None, type=str,
#                     help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
# parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
#                     help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
# parser.add_argument('--eval_only_person', default=False, dest='eval_only_person', action='store_true',
#                     help='Only evaluate on the person class, ignore the other classes.')
# parser.add_argument('--only_last_layer', default=False, dest='only_last_layer', action='store_true',
#                     help='Only train (fine-tune) the last layer.')
# parser.add_argument('--compute_val_loss', default=False, dest='compute_val_loss', action='store_true',
#                     help='Computes the validation loss every according to validation_iter.')

# parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
# args = parser.parse_args()




class Args:
    def __init__(self, config):
        self.batch_size = 5
        self.resume = config.weights
        self.start_iter = 0
        self.num_workers = 4
        self.cuda = True
        self.lr = config.lr
        self.momentum = config.momentum #0.6
        self.decay = config.weight_decay
        self.gamma = None
        self.save_folder = 'yolact_data/train_weights/'
        self.log_folder = 'yolact_data/logs/'
        self.config = config.model
        self.save_interval = config.eval_it #10000
        self.validation_size = config.eval_it #5000
        self.validation_iter = config.eval_it #10000
        self.keep_latest = True
        self.keep_latest_interval = config.eval_it
        self.dataset = None
        self.log = True
        self.log_gpu = False
        self.interrupt = True
        self.batch_alloc = None
        self.autoscale = True
        self.eval_only_person = False
        self.only_last_layer = config.only_last_layer
        self.compute_val_loss = False
        self.max_iter = config.max_it
        self.trained_model = "latest"
        self.shuffle = config.shuffle



# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))

def set_args():
    if args.config is not None:
        set_cfg(args.config)
        cfg.max_iter = args.max_iter

    if args.dataset is not None:
        set_dataset(args.dataset)

    if args.autoscale and args.batch_size != 8:
        factor = args.batch_size / 8
        if __name__ == '__main__':
            print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

        cfg.lr *= factor
        cfg.max_iter //= factor
        cfg.lr_steps = [x // factor for x in cfg.lr_steps]

    
    replace('lr')
    replace('decay')
    replace('gamma')
    replace('momentum')

    # This is managed by set_lr
    cur_lr = args.lr

    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

    if args.batch_size // torch.cuda.device_count() < 6:
        if __name__ == '__main__':
            print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
        cfg.freeze_bn = True

    

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS)) #SSDAugmentation(MEANS)
    
    if args.validation_iter > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS)) #BaseTransform(MEANS)
        test_dataset = COCODetection(image_path=cfg.dataset.test_images,
                                    info_file=cfg.dataset.test_info,
                                    transform=BaseTransform(MEANS)) #BaseTransform(MEANS)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact(only_last_layer=args.only_last_layer)
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, my_config,
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle= args.shuffle, collate_fn=detection_collate,
                                  pin_memory=True)
    
    # data_loader_val = data.DataLoader(val_dataset, args.batch_size,
    #                               num_workers=args.num_workers,
    #                               shuffle=True, collate_fn=detection_collate,
    #                               pin_memory=True)
    
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue
                
            # Stop at the configured number of iterations even if mid-epoch
            if iteration >= cfg.max_iter:
                break
            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration >= cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)
                
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                    
                    
                    wandb.log({#"val_loss": val_losses[idx],
                    "iter":iteration,
                    "total_loss": total
                    })

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            
                    if args.validation_iter > 0:
                        if iteration % args.validation_iter == 0:
                            compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
                            # Added compute validation_loss
                            if args.compute_val_loss:
                                compute_validation_loss(net, data_loader_val, criterion, epoch =epoch, iteration = iteration, log=log if args.log else None)
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        compute_validation_map(epoch, iteration, yolact_net, test_dataset, log if args.log else None, is_test=True)
        # Added compute validation_loss
        if args.compute_val_loss:
            compute_validation_loss(net, data_loader_val, criterion, epoch =epoch, iteration = iteration, log=log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion, epoch, iteration, log:Log=None):
    global loss_types
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }
    with torch.no_grad():
        # Don't switch to eval mode because we want to get losses
        
        print("Computing Validation loss ...")
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            losses = net(datum)

            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])
                 
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)
        print("\n")
        if log is not None:
            precision = 5
            loss_info = {k: round(losses[k].item(), precision) for k in losses}
            loss_info['T'] = round(loss.item(), precision)

            if args.log_gpu:
                log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                            
            log.log('val_loss', loss=loss_info, epoch=epoch, iter=iteration)

            log.log_gpu_stats = args.log_gpu

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None, is_test = False):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)
            if is_test:
                wandb.log({#"val_loss": val_losses[idx],
                        "bbox.AP": val_info['mask']['all'],
                        "segm.AP": val_info['box']['all']
                        })
            else:
                wandb.log({#"val_loss": val_losses[idx],
                        "iter":iteration,
                        "val_ap_box": val_info['mask']['all'],
                        "val_ap_segm": val_info['box']['all']
                        })

        yolact_net.train()

def setup_eval():
    args_list = ['--no_bar', '--max_images='+str(args.validation_size)]
    if args.eval_only_person:
      args_list += ['--only_person']
    eval_script.parse_args(args_list)




### Image visualizer

def predict_images (my_args, n_imgs):
    global args
    args = my_args
    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt(args.save_folder)
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest(args.save_folder, cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    # if args.detect:
    #     cfg.eval_mask_branch = False

    # if args.dataset is not None:
    #     set_dataset(args.dataset)

    with torch.no_grad():
        # if not os.path.exists('results'):
        #     os.makedirs('results')

        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        
        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')
        
        net = net.cuda()

        return evaluate(net, n_imgs)

def evaluate(net:Yolact, train_mode=False, n_imgs = 3):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False
    
    top_k = 15
    th = 0.4
    mask_alpha = 1

    test_imgs = os.listdir(cfg.dataset.test_images)
    
    class_labels = {
        1: "person 1",
        2: "person 2",
        3: "person 3"
        }

    wandb_imgs = []
    for img_path in random.sample(test_imgs, n_imgs):
        img_path = os.path.join(cfg.dataset.test_images, img_path)
        print(img_path)

        # Find predictions for the selected path
        preds = evalimage(net=net, path = img_path)
        
        # Load the image and change it to RGB
        im = cv2.imread(img_path)
        rgb_im = im[:, :, ::-1]

        # Getting image size
        h,w,_ = rgb_im.shape
        
        #Process predictions
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(preds, w, h, visualize_lincomb = False,
                                            crop_masks        = True,
                                            score_threshold   = th)
        cfg.rescore_bbox = save

        idx = t[1].argsort(0, descending=True)[:15]
        #Getting masks,classes scores and boxes from the prediction
        masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        
        # Formatting results into wandb
        bin_mask= np.zeros((h, w))
        box_data = []
        
        class2name = {0:"person"}
        for j in range(classes.shape[0]):
            if scores[j] > th:
                # masks
                sil = masks[j].byte().cpu().numpy()
                av_mask = bin_mask == 0 # Gets the pixels that are not used 
                corrected_sil = np.bitwise_and(av_mask, sil) #Gets the silhouette cropping the overlapped parts
                bin_mask+= corrected_sil * (j+1)

                # scores 
                acc = scores[j]
                acc = round(float(acc), 2)

                #boxes
                x_min, y_min, x_max, y_max = boxes[j]
                x_min, x_max = round(x_min/w, 2), round(x_max/w, 2)
                y_min, y_max = round(y_min/h, 2), round(y_max/h, 2)
                bbox_dict = {"position": {
                                "minX": x_min,
                                "maxX": x_max,
                                "minY": y_min,
                                "maxY": y_max},
                            "class_id" : j+1,
                            "box_caption": "person:{}".format(acc),
                            "scores" : {
                                "acc": acc},
                            }
                box_data.append(bbox_dict)

                #class_labels
                # class_str = class2name[classes[j]]+ " " + str(j+1)
                # class_labels[j+1] = class_str
        wandb_img = wandb.Image(rgb_im, 
        masks={"prediction" : {"mask_data" : bin_mask, "class_labels" : class_labels}},
        boxes={"prediction" : {"box_data" : box_data, "class_labels" : class_labels}})
        wandb_imgs.append(wandb_img)
        # plt.imshow(bin_mask)
        # plt.show()
        # plt.imshow(rgb_im)
        # plt.show()
        # print(box_data)
    return wandb_imgs
        

def evalimage(net:Yolact, path:str):
    input_img = cv2.imread(path)
    frame = torch.from_numpy(input_img).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    return preds


my_config = {
    "batch_size" : 5,
    "resume" : "yolact_data/weights/yolact_plus_base_54_800000.pth",
    "start_iter" : 0,
    "num_workers" : 4,
    "cuda" : True,
    "lr" : 0.0001,
    "momentum" : 0.9, #0.6
    "decay" : None,
    "gamma" : 0.1,
    "save_folder" : 'weights/',
    "log_folder" : 'yolact_data/logs/',
    "config" : "yolact_pp_101_ordered_rc_ucb_gait_config",
    "save_interval" : 1000, #10000
    "validation_size" : 1000, #5000
    "validation_iter" : 1000, #10000
    "keep_latest" : False,
    "keep_latest_interval" : 1000,
    "dataset" : None,
    "log" : True,
    "log_gpu" : False,
    "interrupt" : True,
    "batch_alloc" : None,
    "autoscale" : True,
    "eval_only_person" : False,
    "only_last_layer" : True,
    "compute_val_loss" : False,
    "max_iter" : 2000}

import pprint
import math

#Configurate sweeps

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'segm.AP',
    'goal': 'minimize'   
    }

parameters_dict = {
    'lr_exp': {
        # a flat distribution between 0 and 0.1
        # 'distribution': 'uniform',
        # 'min': -5.7,
        # 'max': -3.7},
        'value': -5},

    'momentum_exp': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': -2,
        'max': -0.3},
    
    # 'momentum_exp': {'value': 0.98},

    'decay_exp': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': -5,
        'max': -3},
    # 'decay_exp': {"value": -4},
    
    
    # 'freeze_at': {
    #     # a flat distribution between 0 and 0.1
    #     'distribution': 'q_uniform',
    #     'min': 1,
    #     'max': 5,
    #     'q': 1},

    # "only_last_layer" :{"values": [0,1]},
    "only_last_layer" :{"value": 0},

    "max_it" :{"value": 2500}, #2500
    "eval_it" : {"value": 500}, #500
    "classes" : {"value":1},
    "model" : {"value":"yolact_pp_101_ordered_rc_ucb_gait_config"},
    "weights" : {"value":"yolact_data/weights/yolact_plus_base_54_800000.pth"},
    "test_th" : {"value":0.4},
    "out_dir" : {"value":"./new_runs/run15"},
    "shuffle": {"value": 0} 
    }
    

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']



def sweep_run(config = None):
    global args
    wandb.init(project="instance_segmentation_train")
    config = wandb.config
    # config.update(my_config) # Used for manual parameters
    
    new_lr = 10**config.lr_exp
    new_momentum = 1 - 10**config.momentum_exp
    new_decay = 10**config.decay_exp
    config.update({"lr":new_lr,
                        "momentum": new_momentum,
                        "weight_decay": new_decay})
    
    # Training weights directory
    out_dir = "{}/{}".format(root_dir, wandb.run.name) #root_dir is global
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    

    args = Args(config)
    args.save_folder = out_dir
    
    set_args()
    print("My args \n")
    pprint.pprint(vars(args))
    os.environ['WANDB_NOTEBOOK_NAME'] = 'yolact_loop.py'
    config.framework = "pytorch"
    # config.out_dir = "./new_runs/{}".format(wandb.run.name)
    train()
    predicted_images = predict_images(args, n_imgs=3)
    wandb.log({"predictions" : predicted_images})
    wandb.run.finish()

if __name__ == '__main__':
    # random.seed(0)
    
    sweep_id = wandb.sweep(sweep_config, project="instance_segmentation_train")     #sweep id will be global
    
    global root_dir
    root_dir = os.path.join("./yolact_runs/sweeps", sweep_id)
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    
    wandb.agent(sweep_id, sweep_run, count=15)
