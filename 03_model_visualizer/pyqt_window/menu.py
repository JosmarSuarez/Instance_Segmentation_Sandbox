from menu_ui import *
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import yolact_visualizer as yol_vis
import os

from utils import timer
from utils.functions import SavePath
from utils.functions import MovingAverage, ProgressBar
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from data import cfg, set_cfg, set_dataset

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from yolact import Yolact

from queue import Queue
from multiprocessing.pool import ThreadPool
from layers.output_utils import postprocess, undo_image_transformation
from data import COCODetection, get_label_map, MEANS, COLORS
from collections import defaultdict

class Args:
  trained_model = '/home/josmar/proyectos/yolact/weights/yolact_pp_101_rc_ucb_gait_19_67000.pth'
  top_k = 15
  cuda = True
  fast_nms = True
  cross_class_nms = False
  display_masks = True
  display_bboxes = True
  display_text = True
  display_scores = True
  display = False
  shuffle = False
  ap_data_file = 'results/ap_data.pkl'
  resume = False
  max_images = -1
  output_coco_json = False
  bbox_det_file = 'results/bbox_detections.json'
  mask_det_file = 'results/mask_detections.json'
  config = None
  output_web_json = False
  web_det_path = 'web/dets/'
  no_bar = False
  display_lincomb = False
  benchmark = False
  no_sort = False
  seed = None
  mask_proto_debug = False
  crop = True
  image = None
  images = None
  # video = "/media/josmar/Nuevo vol/Taller de Grado/new_dataset/UCB walking/VID_20201031_162353.mp4"
  video = "1"
  video_multiframe = 1
  score_threshold = 0.15
  dataset = None
  detect = False
  display_fps = True
  emulate_playback = False
  no_hash = False
  only_mask = False
  
class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

class YolactThread(QThread):
    # notifyProgress = pyqtSignal(int)
    changePixmap = pyqtSignal(QImage, QImage)
    def __init__(self, video_source, parent=None):
        QThread.__init__(self, parent)
        self.video_source = video_source
    
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
        p = convertToQtFormat.scaled(800, 800, Qt.KeepAspectRatio)
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
        self.args=Args()
        self.yolact_init(self.args)

    def yolact_init(self, args):
        if args.config is not None:
            set_cfg(args.config)
        else:
            model_path = SavePath.from_str(args.trained_model)
            # TODO: Bad practice? Probably want to do a name lookup instead.
            args.config = model_path.model_name + '_config'
            print('Config not specified. Parsed %s from the file name.\n' % args.config)
            set_cfg(args.config)

        with torch.no_grad():
            if not os.path.exists('results'):
                os.makedirs('results')

            if args.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            if args.resume and not args.display:
                with open(args.ap_data_file, 'rb') as f:
                    ap_data = pickle.load(f)
                calc_map(ap_data)
                exit()

            if args.image is None and args.video is None and args.images is None:
                dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                            transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
                prep_coco_cats()
            else:
                dataset = None        

            print('Loading model...', end='')
            net = Yolact()
            net.load_weights(args.trained_model)
            net.eval()
            print(' Done.')
            if args.cuda:
                net = net.cuda()
            self.color_cache = defaultdict(lambda: {})
            self.evalvideo(net, args.video)

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb = self.args.display_lincomb,
                                            crop_masks        = self.args.crop,
                                            score_threshold   = self.args.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.args.top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.args.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
            
            if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
                return self.color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    self.color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if self.args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            
            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            
            if self.args.only_mask:
                img_gpu = masks_color_summand
            else:
                img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        elif self.args.only_mask:
            img_gpu = img_gpu * 0

        if self.args.display_fps:
                # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if self.args.display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        if num_dets_to_consider == 0:
            return img_numpy

        if self.args.display_text or self.args.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if self.args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if self.args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if self.args.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
        
        return img_numpy

    def evalvideo(self, net:Yolact, path:str, out_path:str=None):

        net.detect.use_fast_nms = self.args.fast_nms
        net.detect.use_cross_class_nms = self.args.cross_class_nms
        cfg.mask_proto_debug = self.args.mask_proto_debug

        # If the path is a digit, parse it as a webcam index
        is_webcam = path.isdigit()
        
        # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
        cudnn.benchmark = True
        
        if is_webcam:
            vid = cv2.VideoCapture(int(path))
        else:
            vid = cv2.VideoCapture(path)
        
        if not vid.isOpened():
            print('Could not open video "%s"' % path)
            exit(-1)

        target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
        frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if is_webcam:
            num_frames = float('inf')
        else:
            num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        net = CustomDataParallel(net).cuda()
        transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
        frame_times = MovingAverage(100)
        fps = 0
        frame_time_target = 1 / target_fps
        self.running = True
        fps_str = ''
        vid_done = False
        frames_displayed = 0

        if out_path is not None:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
        
        def cleanup_and_exit():
            print("Closing ...")
            pool.terminate()
            vid.release()
            if out_path is not None:
                out.release()
            # exit()

        def get_next_frame(vid):
            frames = []
            for idx in range(self.args.video_multiframe):
                frame = vid.read()[1]
                if frame is None:
                    return frames
                frames.append(frame)
            return frames

        def transform_frame(frames):
            with torch.no_grad():
                frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
                return frames, transform(torch.stack(frames, 0))

        def eval_network(inp):
            with torch.no_grad():
                frames, imgs = inp
                num_extra = 0
                while imgs.size(0) < self.args.video_multiframe:
                    imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                    num_extra += 1
                out = net(imgs)
                if num_extra > 0:
                    out = out[:-num_extra]
                return frames, out

        def prep_frame(inp, fps_str):
            with torch.no_grad():
                frame, preds = inp
                return self.prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

        frame_buffer = Queue()
        video_fps = 0

        # All this timing code to make sure that 
        def play_video():
            try:
                nonlocal frame_buffer, video_fps, is_webcam, num_frames, frames_displayed, vid_done

                video_frame_times = MovingAverage(100)
                frame_time_stabilizer = frame_time_target
                last_time = None
                stabilizer_step = 0.0005
                progress_bar = ProgressBar(30, num_frames)

                while self.running:
                    frame_time_start = time.time()

                    if not frame_buffer.empty():
                        next_time = time.time()
                        if last_time is not None:
                            video_frame_times.add(next_time - last_time)
                            video_fps = 1 / video_frame_times.get_avg()
                        if out_path is None:
                            qt_original = self.convert_to_qt(frame_buffer.get())
                            # qt_rotated = self.convert_to_qt(rotatedImage)
                            self.changePixmap.emit(qt_original, qt_original)

                            # cv2.imshow(path, frame_buffer.get())
                        else:
                            out.write(frame_buffer.get())
                        frames_displayed += 1
                        last_time = next_time

                        if out_path is not None:
                            if video_frame_times.get_avg() == 0:
                                fps = 0
                            else:
                                fps = 1 / video_frame_times.get_avg()
                            progress = frames_displayed / num_frames * 100
                            progress_bar.set_val(frames_displayed)

                            print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                                % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                    
                    # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                    if out_path is None and cv2.waitKey(1) == 27:
                        # Press Escape to close
                        running = False
                    if not (frames_displayed < num_frames):
                        running = False

                    if not vid_done:
                        buffer_size = frame_buffer.qsize()
                        if buffer_size < self.args.video_multiframe:
                            frame_time_stabilizer += stabilizer_step
                        elif buffer_size > self.args.video_multiframe:
                            frame_time_stabilizer -= stabilizer_step
                            if frame_time_stabilizer < 0:
                                frame_time_stabilizer = 0

                        new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                    else:
                        new_target = frame_time_target

                    next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                    target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
                    
                    if out_path is None or self.args.emulate_playback:
                        # This gives more accurate timing than if sleeping the whole amount at once
                        while time.time() < target_time:
                            time.sleep(0.001)
                    else:
                        # Let's not starve the main thread, now
                        time.sleep(0.001)
            except:
                # See issue #197 for why this is necessary
                import traceback
                traceback.print_exc()


        extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

        # Prime the network on the first frame because I do some thread unsafe things otherwise
        print('Initializing model... ', end='')
        first_batch = eval_network(transform_frame(get_next_frame(vid)))
        print('Done.')

        # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
        sequence = [prep_frame, eval_network, transform_frame]
        pool = ThreadPool(processes=len(sequence) + self.args.video_multiframe + 2)
        pool.apply_async(play_video)
        active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

        print()
        if out_path is None: print('Press Escape to close.')
        try:
            while vid.isOpened() and self.running:
                # Hard limit on frames in buffer so we don't run out of memory >.>
                while frame_buffer.qsize() > 100:
                    time.sleep(0.001)

                start_time = time.time()

                # Start loading the next frames from the disk
                if not vid_done:
                    next_frames = pool.apply_async(get_next_frame, args=(vid,))
                else:
                    next_frames = None
                
                if not (vid_done and len(active_frames) == 0):
                    # For each frame in our active processing queue, dispatch a job
                    # for that frame using the current function in the sequence
                    for frame in active_frames:
                        _args =  [frame['value']]
                        if frame['idx'] == 0:
                            _args.append(fps_str)
                        frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                    
                    # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                    for frame in active_frames:
                        if frame['idx'] == 0:
                            frame_buffer.put(frame['value'].get())

                    # Remove the finished frames from the processing queue
                    active_frames = [x for x in active_frames if x['idx'] > 0]

                    # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                    for frame in list(reversed(active_frames)):
                        frame['value'] = frame['value'].get()
                        frame['idx'] -= 1

                        if frame['idx'] == 0:
                            # Split this up into individual threads for prep_frame since it doesn't support batch size
                            active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
                            frame['value'] = extract_frame(frame['value'], 0)
                    
                    # Finish loading in the next frames and add them to the processing queue
                    if next_frames is not None:
                        frames = next_frames.get()
                        if len(frames) == 0:
                            vid_done = True
                        else:
                            active_frames.append({'value': frames, 'idx': len(sequence)-1})

                    # Compute FPS
                    frame_times.add(time.time() - start_time)
                    fps = self.args.video_multiframe / frame_times.get_avg()
                else:
                    fps = 0
                
                fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (fps, video_fps, frame_buffer.qsize())
                if not self.args.display_fps:
                    print('\r' + fps_str + '    ', end='')

        except KeyboardInterrupt:
            print('\nStopping...')
        
        cleanup_and_exit()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        MainWindow.setWindowTitle(self,"Model Visualizer")
        
        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.setEnabled(False)

        self.saveButton.clicked.connect(self.save_file)
        
    

    def start_video(self):
        
        self.video_thread = YolactThread(video_source=1)
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.start()

        self.label_video.setHidden(False)
        self.label_mask.setHidden(False)

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def stop_video(self):
        self.video_thread.running = False
        self.label_video.setHidden(True)
        self.label_mask.setHidden(True)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
    
        """
        setImage(self, image1, image2)
        
        Receives two QImages via pyqtSignal and shows them in the screen  
        """
    @pyqtSlot(QImage, QImage)
    def setImage(self, image1, image2):
        self.label_video.setPixmap(QPixmap.fromImage(image1))
        self.label_mask.setPixmap(QPixmap.fromImage(image2))
    
    def save_file(self):
        _dir = QFileDialog.getSaveFileName(self, 'Save File')
        self.label_save.setText(_dir[0])
        
    # def actualizar(self):
    #     self.label.setText("¡Acabas de hacer clic en el botón!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()






