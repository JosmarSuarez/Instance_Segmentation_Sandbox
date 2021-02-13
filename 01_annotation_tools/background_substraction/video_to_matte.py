#!/usr/bin/env python
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Importing libraries

# %%
# folder and video to images dependencies
import os
import subprocess as sp
import shutil

# Segmentation dependencies
import sys
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

import numpy as np
from PIL import Image
import cv2, pdb, glob, argparse

import tensorflow as tf

#importing numba to clean all memory allocated
from numba import cuda

#Matting dependencies
# from __future__ import print_function


import glob, time, argparse, pdb
#import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from functions import *
from networks import ResnetConditionHR

# Import time to measure time of execution
import time

# Import json to write times
import json
#Tensorflow doesnt release memory after executed(NO used)
import multiprocessing
import re

# %% [markdown]
# ## Separating video in images

# %%
def video2image(input_video, size, input_folder):
    FFMPEG_BIN = "ffmpeg"
    command = [ FFMPEG_BIN,
                '-i', input_video, 
                '-vf', 'scale={}:{}'.format(size[0],size[1]),
                os.path.join(input_folder,"%04d_img.png")]
    print("Extracting frames from video into /input folder...")
    sp.run(command)
    print("Extraction finished successfully")

# %% [markdown]
# ## Running segmentation
# Using deeplabv3 to segment the images

# %%
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        #"""Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        #I added this code
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #----------------
        self.sess = tf.Session(config=config,graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


# %%
def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
          colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


# %%
import shutil
def deeplab_segmentation(input_folder, include_empty=True, total_frames = -1, back_img_path=None):
    dir_name = input_folder
    ## setup ####################

    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


    MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

    model_dir = 'deeplab_model'
    if not os.path.exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
        print('downloading model to %s, this might take a while...' % download_path)
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], 
                        download_path)
        print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')

    #######################################################################################

    list_im=glob.glob(dir_name + '/*_img.png'); list_im.sort()
    #Creating useful frame counter
    if total_frames == -1:
       total_frames = len(list_im)

    count = 0
    # variables to create back img
    count_empty = 0
    isCreated = False
    #--------------------- 
    for i in range(0,len(list_im)):
        
        if count >= total_frames:
            break 
        image = Image.open(list_im[i])

        res_im,seg=MODEL.run(image)

        seg=cv2.resize(seg.astype(np.uint8),image.size)

        mask_sel=(seg==15).astype(np.float32)

        name=list_im[i].replace('img','masksDL')
        out_mask = (255*mask_sel).astype(np.uint8)
        #opening
        kernel = np.ones((5,5),np.uint8)
        out_mask_open = cv2.morphologyEx(out_mask, cv2.MORPH_OPEN, kernel)
        
        if(cv2.countNonZero(out_mask_open)>15 or include_empty):
            cv2.imwrite(name, out_mask)
            count+=1
            count_empty=0 #reset back_img_counter
        else:
            count_empty+=1 #back_img_counter
            if(count_empty > 8 and not isCreated and back_img_path != None):
                shutil.copy(list_im[i], back_img_path)
                isCreated = True
                print('\tbackground image extracted')
        sys.stdout.write('\r'+  "Done: {}/{}    Frames: {}/{}".format(i+1,len(list_im), count, total_frames))
        sys.stdout.flush()
    str_msg='\nDone: ' + dir_name
    print(str_msg)


# %%
# image = Image.open('/home/josmar/proyectos/Background-Matting/walking_clips/temp/1280x720/input/0001_img.png')
# # image.save("walking_clips/back_mario_1280x720.png")
# shutil.copy('/home/josmar/proyectos/Background-Matting/walking_clips/temp/1280x720/input/0001_img.png',"walking_clips/back_mario_1280x720.png")


# %%
#Cleaning the allocated memory
def clean_memory():
    device = cuda.get_current_device()
    device.close()

# %% [markdown]
# ## Obtaining the background matte

# %%
def obtain_matting(input_folder, output_folder, back_path, trained_model, include_empty=True, total_frames = -1):    
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])


    # """Parses arguments."""
    # parser = argparse.ArgumentParser(description='Background Matting.')
    # parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',choices=['real-fixed-cam', 'real-hand-held', 'syn-comp-adobe'],help='Trained background matting model')
    # parser.add_argument('-o', '--output_dir', type=str, required=True,help='Directory to save the output results. (required)')
    # parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to load input images. (required)')
    # parser.add_argument('-tb', '--target_back', type=str,help='Directory to load the target background.')
    # parser.add_argument('-b', '--back', type=str,default=None,help='Captured background image. (only use for inference on videos with fixed camera')


    # args=parser.parse_args()

    #input model
    model_main_dir='Models/' + trained_model + '/';
    #input data path
    data_path= input_folder

    is_video=True
    print('Using video mode')


    #initialize network
    fo=glob.glob(model_main_dir + 'netG_epoch_*')
    model_name1=fo[0]
    netM=ResnetConditionHR(input_nc=(3,3,1,4),output_nc=4,n_blocks1=7,n_blocks2=3)
    netM=nn.DataParallel(netM)
    netM.load_state_dict(torch.load(model_name1))
    netM.cuda(); netM.eval()
    cudnn.benchmark=True
    reso=(512,512) #input reoslution to the network

    #load captured background for video mode, fixed camera
    if back_path is not None:
        bg_im0=cv2.imread(back_path); bg_im0=cv2.cvtColor(bg_im0,cv2.COLOR_BGR2RGB);


    
    #Create a list of test masks (changed to read only segmentation masks)
    test_masks = [f for f in os.listdir(data_path) if
                os.path.isfile(os.path.join(data_path, f)) and f.endswith('_masksDL.png')]
    #Create a list of test images
    test_imgs = [name.replace('_masksDL','_img') for name in test_masks]

    test_imgs.sort()
    test_masks.sort()
    
    #Limit the number of frames
    if total_frames ==-1:
        total_frames = len(test_imgs)

    #output directory
    result_path= output_folder

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    #Creating number of frames processed counter
    count= 0

    for i in range(0,len(test_masks)):
        
        #Close the loop when the number of frames desired is reached
        if count >= total_frames:
            break
        
        #read image
        filename = test_imgs[i]    
        
        #original image
        bgr_img = cv2.imread(os.path.join(data_path, filename)); bgr_img=cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB);

        if back_path is None:
            #captured background image
            bg_im0=cv2.imread(os.path.join(data_path, filename.replace('_img','_back'))); bg_im0=cv2.cvtColor(bg_im0,cv2.COLOR_BGR2RGB);

        #segmentation mask
        rcnn = cv2.imread(os.path.join(data_path, filename.replace('_img','_masksDL')),0);

        if is_video: #if video mode, load target background frames
            #target background path
            # back_img10=cv2.imread(os.path.join(args.target_back,filename.replace('_img.png','.png'))); back_img10=cv2.cvtColor(back_img10,cv2.COLOR_BGR2RGB);
            #Green-screen background
            back_img20=np.zeros(bgr_img.shape); back_img20[...,0]=120; back_img20[...,1]=255; back_img20[...,2]=155;

            #create multiple frames with adjoining frames
            gap=20
            multi_fr_w=np.zeros((bgr_img.shape[0],bgr_img.shape[1],4))
            idx=[i-2*gap,i-gap,i+gap,i+2*gap]
            for t in range(0,4):
                if idx[t]<0:
                    idx[t]=len(test_imgs)+idx[t]
                elif idx[t]>=len(test_imgs):
                    idx[t]=idx[t]-len(test_imgs)

                file_tmp=test_imgs[idx[t]]
                bgr_img_mul = cv2.imread(os.path.join(data_path, file_tmp));
                multi_fr_w[...,t]=cv2.cvtColor(bgr_img_mul,cv2.COLOR_BGR2GRAY);

        else:
            ## create the multi-frame
            multi_fr_w=np.zeros((bgr_img.shape[0],bgr_img.shape[1],4))
            multi_fr_w[...,0] = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY);
            multi_fr_w[...,1] = multi_fr_w[...,0]
            multi_fr_w[...,2] = multi_fr_w[...,0]
            multi_fr_w[...,3] = multi_fr_w[...,0]

            
        #crop tightly
        bgr_img0=bgr_img;
        isBlack= (cv2.countNonZero(rcnn)==0)
        if(cv2.countNonZero(rcnn)>15):#Edited
            #counter of processed frames
            count+=1

            bbox=get_bbox(rcnn,R=bgr_img0.shape[0],C=bgr_img0.shape[1])

            # crop_list=[bgr_img,bg_im0,rcnn,back_img10,back_img20,multi_fr_w]
            crop_list=[bgr_img,bg_im0,rcnn,back_img20,multi_fr_w]
            crop_list=crop_images(crop_list,reso,bbox)
            bgr_img=crop_list[0]; bg_im=crop_list[1]; rcnn=crop_list[2]; back_img2=crop_list[3]; multi_fr=crop_list[4]
            # bgr_img=crop_list[0]; bg_im=crop_list[1]; rcnn=crop_list[2]; back_img1=crop_list[3]; back_img2=crop_list[4]; multi_fr=crop_list[5]

            #process segmentation mask
            kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            rcnn=rcnn.astype(np.float32)/255; rcnn[rcnn>0.2]=1;
            K=25

            zero_id=np.nonzero(np.sum(rcnn,axis=1)==0)
            del_id=zero_id[0][zero_id[0]>250]
            if len(del_id)>0:
                del_id=[del_id[0]-2,del_id[0]-1,*del_id]
                rcnn=np.delete(rcnn,del_id,0)
            rcnn = cv2.copyMakeBorder( rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)


            rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
            rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
            rcnn=cv2.GaussianBlur(rcnn.astype(np.float32),(31,31),0)
            rcnn=(255*rcnn).astype(np.uint8)
            rcnn=np.delete(rcnn, range(reso[0],reso[0]+K), 0)


            #convert to torch
            img=torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0); img=2*img.float().div(255)-1
            bg=torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0); bg=2*bg.float().div(255)-1
            rcnn_al=torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0); rcnn_al=2*rcnn_al.float().div(255)-1
            multi_fr=torch.from_numpy(multi_fr.transpose((2, 0, 1))).unsqueeze(0); multi_fr=2*multi_fr.float().div(255)-1


            with torch.no_grad():
                img,bg,rcnn_al, multi_fr =Variable(img.cuda()),  Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(multi_fr.cuda())
                input_im=torch.cat([img,bg,rcnn_al,multi_fr],dim=1)
                
                alpha_pred,fg_pred_tmp=netM(img,bg,rcnn_al,multi_fr)
                
                al_mask=(alpha_pred>0.95).type(torch.cuda.FloatTensor)

                # for regions with alpha>0.95, simply use the image as fg
                fg_pred=img*al_mask + fg_pred_tmp*(1-al_mask)

                alpha_out=to_image(alpha_pred[0,...]); 

                #refine alpha with connected component
                labels=label((alpha_out>0.05).astype(int))
                try:
                    assert( labels.max() != 0 )
                except:
                    continue
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                alpha_out=alpha_out*largestCC

                alpha_out=(255*alpha_out[...,0]).astype(np.uint8)                

                fg_out=to_image(fg_pred[0,...]); fg_out=fg_out*np.expand_dims((alpha_out.astype(float)/255>0.01).astype(float),axis=2); fg_out=(255*fg_out).astype(np.uint8)

                #Uncrop
                R0=bgr_img0.shape[0];C0=bgr_img0.shape[1]
                alpha_out0=uncrop(alpha_out,bbox,R0,C0)
                fg_out0=uncrop(fg_out,bbox,R0,C0)

                #it was down else before to include both states (*)
                comp_im_tr2=composite4(fg_out0,back_img20,alpha_out0)

                cv2.imwrite(result_path+'/'+filename.replace('_img','_out'), alpha_out0)
                cv2.imwrite(result_path+'/'+filename.replace('_img','_fg'), cv2.cvtColor(fg_out0,cv2.COLOR_BGR2RGB))
                # cv2.imwrite(result_path+'/'+filename.replace('_img','_compose'), cv2.cvtColor(comp_im_tr1,cv2.COLOR_BGR2RGB))
                cv2.imwrite(result_path+'/'+filename.replace('_img','_matte').format(i), cv2.cvtColor(comp_im_tr2,cv2.COLOR_BGR2RGB))
        else:#(*)
            alpha_out0=rcnn
            fg_out0=bgr_img0
            
            if(include_empty):
                comp_im_tr2=composite4(fg_out0,back_img20,alpha_out0)

                cv2.imwrite(result_path+'/'+filename.replace('_img','_out'), alpha_out0)
                cv2.imwrite(result_path+'/'+filename.replace('_img','_fg'), cv2.cvtColor(fg_out0,cv2.COLOR_BGR2RGB))
                # cv2.imwrite(result_path+'/'+filename.replace('_img','_compose'), cv2.cvtColor(comp_im_tr1,cv2.COLOR_BGR2RGB))
                cv2.imwrite(result_path+'/'+filename.replace('_img','_matte').format(i), cv2.cvtColor(comp_im_tr2,cv2.COLOR_BGR2RGB))


        #compose
        # back_img10=cv2.resize(back_img10,(C0,R0)); back_img20=cv2.resize(back_img20,(C0,R0))
        # comp_im_tr1=composite4(fg_out0,back_img10,alpha_out0)

        # print("fg_out0",fg_out0.shape)
        # print("back_img20",back_img20.shape)
        # print("alpha_out0",alpha_out0.shape)
        
        
        

        sys.stdout.write('\r'+  "Done: {}/{}    Frames: {}/{}".format(i+1,len(test_imgs), count, total_frames))
        sys.stdout.flush()
    print('\n')
        


# %%
# os.path.join(workspace,'teaser_matte.mp4')
def images2video(images_folder, video_folder):
    FFMPEG_BIN = "ffmpeg"
    command = [ FFMPEG_BIN,
                '-f', "image2",
                '-r', '30',
                '-i', os.path.join(images_folder,"%04d_matte.png"),
                '-vcodec', 'libx264',
                '-crf', '15',
                '-pix_fmt', 'yuv420p' ,
                video_folder]
    print("Extracting frames from /output folder...")
    sp.run(command)
    print("Extraction finished successfully")


# %%
def save_dic(json_var,filename):
  with open(filename, 'w') as json_file:
    json.dump(json_var, json_file)

# %% [markdown]
# ## Generate video

# %%
import cv2
import numpy as np
import glob
import re
def generate_video(original_folder,processed_folder, media_folder):
    out_images = glob.glob(os.path.join(processed_folder,'*_out.png'))
    pattern = 'output/(.*?)_out'
    frame_list = [re.search(pattern, name).group(1) for name in out_images]
    frame_list.sort()

    video_data = {}
    video_data[os.path.join(media_folder, "trimmed.mp4")] = os.path.join(original_folder, "{}_img.png")##Check the keys
    video_data[os.path.join(media_folder,"masks.mp4")] = os.path.join(original_folder, "{}_masksDL.png")
    video_data[os.path.join(media_folder,"out.mp4")] = os.path.join(processed_folder, "{}_out.png")
    video_data[os.path.join(media_folder,"fg.mp4")] = os.path.join(processed_folder, "{}_fg.png")
    video_data[os.path.join(media_folder,"matte.mp4")] = os.path.join(processed_folder, "{}_matte.png")
    for v in video_data:
        print('\ngenerating ', v)
        img_array = []
        vid_images = [video_data[v].format(f) for f in frame_list]
        

        for filename in vid_images:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        
        out = cv2.VideoWriter(v,cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
            sys.stdout.write('\r'+  "Done: {}/{}".format(i+1,len(img_array)))
            sys.stdout.flush()
        out.release()

# %% [markdown]
# ## Main Program
# %% [markdown]
# ## Example of use
# ```python
# sizes = [(1280,720)]#If you want to do it to multiple sizes put the size in a tuple (w,h)
# time_dic = {}
# desired_frames = 300
# # Model
# trained_model = "real-hand-held"
# # Folder that contains all the inputs and where the output will be saved
# workspace = "walking_clips/temp"
# # Base video
# input_video = "walking_clips/mario.mp4"
# # Background image
# back_path = "walking_clips"
# # Results folder
# results_folder = "walking_clips/results"
# process_video(workspace, trained_model, sizes, input_video, back_path, desired_frames, results_folder)
# ```


# %% [markdown]
# ## For only one video

# %%
def process_video(workspace, trained_model, sizes, input_video, desired_frames, results_folder, back_path=None, include_empty=True):
    w_time = {}
    print('inside_function\n\n',sizes)
    for size in sizes:
        print("\n\n Folder:",size)
        size_str = '{}x{}'.format(size[0], size[1])    
        w_time[size_str] = []

        # Creating folder paths
        input_folder = os.path.join(workspace,size_str,"input")
        output_folder = os.path.join(workspace,size_str, "output")

        # Creating the folders
        size_folder = os.path.join(workspace,size_str)
        res_size_folder = os.path.join(results_folder, size_str)
             

        folders = [workspace, size_folder, input_folder, output_folder, results_folder, res_size_folder]
        for folder in folders:
            if(os.path.exists(folder)):
                print("Folder '{}'\talready exists, skipping ...".format(folder))
            else:
                print("Folder '{}'\tnot found, creating one ...".format(folder))
                os.mkdir(folder)
        print("Folders succesfully created")

        #converting videos into images
        video2image(input_video, size, input_folder)
        #Segmentating the video
        start = time.perf_counter()
        deeplab_segmentation(input_folder, include_empty=include_empty, total_frames = desired_frames, back_img_path=back_path)        
        # process = multiprocessing.Process(deeplab_segmentation(input_folder))
        # process.start()
        # process.join()
        end = time.perf_counter()
        
        
        w_time[size_str].append(end-start)
        print("finished")
        #Cleaning the tensorflow allocated memory

        start2 = time.perf_counter()
        
        obtain_matting(input_folder, output_folder, back_path, trained_model, include_empty=include_empty, total_frames = desired_frames)
        

        # process = multiprocessing.Process(obtain_matting(input_folder, output_folder, back_path, trained_model))
        # process.start()
        # process.join()

        end2 = time.perf_counter()
        #Getting the video matte
        # start = time.time()
        # obtain_matting(input_folder, output_folder, back_path, trained_model)
        # end = time.time()
        
        w_time[size_str].append(end2-start2)
        generate_video(input_folder,output_folder,res_size_folder)
        shutil.rmtree(size_folder)
    return(w_time)


# %%
#Datos de prueba:
# trained_model real-hand-held
# input_video /home/josmar/proyectos/codes/annotation_tools/background_substraction/output.mp4
# back_path pruebas/back_image.png
# desired_frames 300
# sizes [(1280,720)]
# workspace pruebas
# results_folder pruebas/results 
# -m real-hand-held -i /home/josmar/proyectos/codes/annotation_tools/background_substraction/output.mp4 -b pruebas/back_image.png -f 100 -s [(1280,720)] -w pruebas -r pruebas/results


parser = argparse.ArgumentParser(description='Background matting')
parser.add_argument('-m', '--trained_model', type=str, required=True,help='Model to be used to extract the matting (real-hand-held or real-fixed-cam )(required)')
parser.add_argument('-i', '--input_video', type=str, required=True,help='Path to video to which matting will be applied. (required)')
parser.add_argument('-b', '--back_path', type=str, required=True,help='Path to video background image.(required)')
parser.add_argument('-f', '--desired_frames', type=int, required=False, default=-1,help='Numbers of frames required to extract (optional).')
parser.add_argument('-s', '--size', type=str, required=True, nargs='+', help='List of sizes in tuple format (Example: -s 1280x720) (required)')
parser.add_argument('-w', '--workspace', type=str, required=True,help='Folder where temp files will be written (required)')
parser.add_argument('-r', '--results_folder', type=str, required=True,help='Folder where the results will be written (required)')
parser.add_argument('-e', '--include_empty', action="store_true", help='Includes empty background images in the final result')
args=parser.parse_args()
# Model
trained_model = args.trained_model
# Sizes to be processed
sizes = args.size#If you want to do it to multiple sizes put the size in a tuple (w,h)

sizes = [size.split('x') for size in sizes]
sizes =[list(map(int, size)) for size in sizes]


# Dictionary where times will be stored
time_dic = {}
# temp file where images will be saved
workspace = args.workspace
# Input video
input_video= args.input_video 
# Number of frames at output
desired_frames = args.desired_frames
# Folder that will contain the results
results_folder=args.results_folder
# Folder that will contain the background images
back_path = args.back_path
# Boolean to decide to save empty frames
include_empty = args.include_empty
if(include_empty):
    print("My variable is true")
else:
    print("My variable is false")
print("\n\n\n\n\n")

process_video(workspace, trained_model, sizes, input_video, desired_frames, results_folder, back_path, include_empty)

# %%



