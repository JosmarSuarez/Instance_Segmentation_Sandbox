import yolact_visualizer as yol_vis

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

args = Args()
yol_vis.run(args)