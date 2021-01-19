import sys

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

#TODO : this is a temporary expedient

sys.path.append('/home/josmar/proyectos/centermask2')
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from .predictor import VisualizationDemo
from centermask.config import get_cfg

class CentermaskArgs:
    config_file = "/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/centermask2_files/configs/centermask/centermask_lite_V_19_eSE_FPN_ms_4x.yaml"
    webcam =  1 # 'Take inputs from webcam.'
    video_input = None #"/media/josmar/Nuevo vol/Dataset Casia/DatasetB-1/video/001-bg-01-000.avi" #'Path to video file.'
    input =   None #'A list of space separated input images'
    output =  None#'A file or directory to save output visualizations. '
    confidence_threshold = 0.4#0.4
    opts = ["MODEL.WEIGHTS","/home/josmar/proyectos/codes/02_detectron2_notebooks/P4_centermask_sil_val/output/run11_centermask2_2nd_method/model_final.pth",
    "MODEL.FCOS.NUM_CLASSES", "1"]

    show_image = True
    show_boxes=True
    show_labels=True
    set_alpha=0.7
    img_binary = True
    size = None



class CentermaskThread(QThread):
    # notifyProgress = pyqtSignal(int)
    changePixmap = pyqtSignal(QImage, QImage)
    changeFPS = pyqtSignal(str)
    def __init__(self, args, parent=None):
        QThread.__init__(self, parent)
        self.args = args
    
    def set_res(self, cap, x,y):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) 
    
    def convert_to_qt(self, rgbImage):
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        return p

    def run(self):
        
        # self.running = True
        # cap = cv2.VideoCapture(self.video_source)
        # self.set_res(cap, 1280, 720)

        # # Check if camera opened successfully 
        # if (cap.isOpened()== False):  
        #     print("Error opening video  file") 
        
        # # Read until video is completed 
        # while(cap.isOpened() and self.running): 
        #     ret, frame = cap.read()
        #     if ret:
        #         # https://stackoverflow.com/a/55468544/6622587
        #         # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         rotatedImage = cv2.flip(rgbImage, 1)

        #         qt_original = self.convert_to_qt(rgbImage)
        #         qt_rotated = self.convert_to_qt(rotatedImage)
        #         self.changePixmap.emit(qt_original, qt_rotated)

        #     # Break the loop 
        #     else:  
        #         break
        pass
        self.centermask_init(self.args)

    def centermask_init(self, args):
        mp.set_start_method("spawn", force=True)
        # args = get_parser().parse_args()
        # args = CentermaskArgs()
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = self.setup_cfg(args)

        demo = VisualizationDemo(cfg, args)
        self.running=True

        if args.input:
            if os.path.isdir(args.input):
                args.input = [os.path.join(args.input, fname) for fname in os.listdir(args.input)]
                print (args.input)
            elif len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()

                if args.img_binary:
                    out_img = demo.run_on_image(img)
                    img_name = os.path.basename(path).split(".")[0] + ".png"
                    out_filename = os.path.join(args.output, img_name)
                    cv2.imwrite(out_filename, out_img)

                else:
                    predictions, visualized_output = demo.run_on_image(img)
                    logger.info(
                        "{}: detected {} instances in {:.2f}s".format(
                            path, len(predictions["instances"]), time.time() - start_time
                        )
                    )

                    if args.output:
                        if os.path.isdir(args.output):
                            assert os.path.isdir(args.output), args.output
                            out_filename = os.path.join(args.output, os.path.basename(path))
                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.output
                        visualized_output.save(out_filename)
                    else:
                        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit
        elif args.webcam is not None:
            assert args.input is None, "Cannot have both --input and --webcam!"
            cam = cv2.VideoCapture(args.webcam)
            v_w, v_h = self.args.size.split("x")  
            self.set_res(cam, v_w, v_h)

            # for frame, vis in tqdm.tqdm(demo.run_on_video(cam)):
            for frame, vis, fps in demo.run_on_video(cam):
                # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # cv2.imshow(WINDOW_NAME, vis)
                
                rgbOut = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                qt_masked = self.convert_to_qt(rgbOut)
                qt_original = self.convert_to_qt(frame)
                self.changePixmap.emit(qt_original, qt_masked)
                text_fps = "FPS: {}".format(fps)
                self.changeFPS.emit(text_fps)

                if not self.running:
                    break  # esc to quit
            cam.release()
            cv2.destroyAllWindows()
            
        elif args.video_input:
            video = cv2.VideoCapture(args.video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(args.video_input)

            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + ".mkv"
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    # fourcc=cv2.VideoWriter_fourcc(*"x264"),
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(args.video_input)
            for frame, vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                if args.output:
                    output_file.write(vis_frame)
                else:
                    # cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    # cv2.imshow(basename, vis_frame)

                    rgbOut = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    qt_masked = self.convert_to_qt(rgbOut)
                    qt_original = self.convert_to_qt(frame)
                    self.changePixmap.emit(qt_original, qt_masked)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                if not self.running:
                    break  # esc to quit
            video.release()
            if args.output:
                output_file.release()
            else:
                cv2.destroyAllWindows()

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.freeze()
        return cfg

    
