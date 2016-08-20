# Face Detection with End-to-End Integration of a ConvNet and a 3D Model

## Reproducing all experimental results in the paper
Yunzhu Li, Benyuan Sun, Tianfu Wu and Yizhou Wang, "Face Detection with End-to-End Integration of a ConvNet and a 3D Model", ECCV 2016 (https://arxiv.org/abs/1606.00850)

The code is mainly written by Y.Z. Li (leo.liyunzhu@pku.edu.cn) and B.Y. Sun (sunbenyuan@pku.edu.cn). Please feel free to report issues to him. 

The code is based on the mxnet package (https://github.com/dmlc/mxnet/). 

If you find the code is useful in your projects, please consider to cite the paper,

@inproceedings{FaceDetection-ConvNet-3D,
  author    = {Yunzhu Li and Benyuan Sun and Tianfu Wu and Yizhou Wang},
  title     = {Face Detection with End-to-End Integration of a ConvNet and a 3D Model},
  booktitle = {ECCV},
  year      = {2016}
}


## Compile
Please refer to https://github.com/dmlc/mxnet/ on how to compile

## Prepare training data
Download AFLW datset and generate a list for the training data in the form of:
ID file_path width height resize_factor number_of_faces [a list of information of each faces]

The information of different faces should be seperated by space and in the form:
x y width height(of bounding box) x y width height(of projected bounding box) number_of_keypoints [keypoint_name keypoint_x keypoint_y projected_keypoint_x projected_keypoint_y](for every keypoint) ellipse_x ellipse_y ellipse_radius ellipse_minoraxes ellipse_majoraxes [9 parameters of scale * rotation matrix] [3 translation parameters]

Note: projected information is not used now, so it can be replaces by any number

## training procedure
1. run Path_To_The_Code/ALFW/vgg16_rpn.py
2. To finetune on FDDB dataset, run Path_To_The_Code/ALFW/fddb_finetune.py

## prediction procedure
AFW: run Path_To_The_Code/afw_predict.py
FDDB: run Path_To_The_Code/predict_final.py