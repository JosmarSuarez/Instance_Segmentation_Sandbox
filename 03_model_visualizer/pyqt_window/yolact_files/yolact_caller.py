import yolact_visualizer as yol_vis
import timeit

class Args:
  # trained_model = '/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/weights/yolact_pp_101_rc_ucb_gait_4_17000.pth'
  trained_model = '/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/yolact_files/weights/yolact_pp_101_ordered_rc_ucb_gait_25_87000.pth'
  top_k = 1
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
  video = None
  video_multiframe = 1
  score_threshold = 0.05
  dataset = None
  detect = False
  display_fps = True
  emulate_playback = False
  no_hash = False
  only_mask = True

input_folder = "/media/josmar/Nuevo vol/Taller de Grado/new_dataset/Casia_processed_silhouettes/Casia_B_90_images"
output_folder = "/media/josmar/Nuevo vol/Taller de Grado/new_dataset/Casia_processed_silhouettes/yolact_pp_101_ordered_rc_ucb_gait_25_87000-lr0001"



args = Args()
args.images = "{}:{}".format(input_folder, output_folder)
args.display_bboxes = False
args.display_text = False
args.display_scores = False

t_start = timeit.timeit()

yol_vis.run(args)

t_end = timeit.timeit()

print("Elapsed time: ", t_end-t_start)