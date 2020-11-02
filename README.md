# Instance_Segmentation_Sandbox
It's a repository to test my instance segmentation codes in more devices
## Useful Tools (01_annotation_tools)
### /01_annotation_tools/background_substraction/video_to_matte.py
This tools allows the user to perform background matting to their videos and returns the semantic segmentation, matte and green screen in mp4 format using the technique presented in [[1]](#1) which code is available at https://github.com/senguptaumd/Background-Matting.


![Background_matting](github_media/background_substraction/gif/background_matting.GIF)
#### Usage 
Here is one example of usage of the code:
```Bash
python video_to_matte.py -m real-hand-held -i /home/josmar/proyectos/codes/annotation_tools/background_substraction/output.mp4 -b pruebas/back_image.png -f 300 -s "1280x720" -w pruebas -r pruebas/results
```
The args of the program are the following:
```
Background matting (args)

optional arguments:
  -h, --help            show this help message and exit
  -m TRAINED_MODEL, --trained_model TRAINED_MODEL
                        Model to be used to extract the matting (real-hand-
                        held or real-fixed-cam )(required)
  -i INPUT_VIDEO, --input_video INPUT_VIDEO
                        Path to video to which matting will be applied.
                        (required)
  -b BACK_PATH, --back_path BACK_PATH
                        Path to video background image.(required)
  -f DESIRED_FRAMES, --desired_frames DESIRED_FRAMES
                        Numbers of frames required to extract (optional).
  -s SIZE [SIZE ...], --size SIZE [SIZE ...]
                        List of sizes in tuple format (Example: -s 1280x720)
                        (required)
  -w WORKSPACE, --workspace WORKSPACE
                        Folder where temp files will be written (required)
  -r RESULTS_FOLDER, --results_folder RESULTS_FOLDER
                        Folder where the results will be written (required)
  -e, --include_empty   Includes empty background images in the final result
  ### video_to_matte.py
```
### /01_annotation_tools/background_substraction/opencv_operations.ipynb
This is a jupyter notebook that creates a video or folder applying morpholical operations to videos
### /01_annotation_tools/bit_to_polyV3.ipynb
This jupyter notebook creates annotation files in COCO format (Representing the segmentations as polygons) using a folder containing mask annotations as input
### /01_annotation_tools/bit_to_RLE.ipynb
This jupyter notebook creates annotation files in COCO format (Representing the segmentations as Run Length encoding) using a folder containing mask annotations as input
### /01_annotation_tools/image_cropper.ipynb
Converts rectangular images into square images using the bounding boxes from the coco annotations to capture the instance
### /01_annotation_tools/image_resize.ipynb
Resizes a folder that contains images
## References
<a id="1">[1]</a> 
S. Sengupta, V. Jayaram, B. Curless. (2020). 
Background Matting: The World is Your Green Screen. 
Computer Vision and Pattern Regognition (CVPR).
