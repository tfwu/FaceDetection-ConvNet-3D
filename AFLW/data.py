import numpy as np
import mxnet as mx
import random
from PIL import Image, ImageFilter
from PIL import ImageFile
import cv2
import sys, os
from mxnet.io import DataIter
from symbol_vgg16 import get_vgg16_rpn
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True


class position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

    def in_box(self, a, b):
        return a >= self.x and a < self.x + self.width and \
               b >= self.y and b < self.y + self.height


class feature_pos:
    def __init__(self, arr):
        self.x, self.y, self.pts_x, self.pts_y = [int(float(item)) for item in arr]


class Face:
    def __init__(self):
        self.num_keypoint = 0
        self.bbox = position()
        self.proj_bbox = position()
        self.keypoints = {}
        self.ellipse = []


class myImage:
    def __init__(self):
        self.filename = ''
        self.width = 0
        self.height = 0
        self.image = None
        self.faces = {}
        self.num_faces = 0
        self.resize_factor = 0


class FileIter(DataIter):
    def __init__(self, file_lst, class_names, file_mface,
                 rgb_mean=(117, 117, 117),
                 data_name="data",
                 cls_label_name="cls_label",
                 proj_label_name="proj_label",
                 proj_weight_name="proj_weight",
                 ground_truth_name="ground_truth",
                 bbox_label_name="bbox_label",
                 bbox_weight_name="bbox_weight",
                 bgfg=False,
                 num_data_used=-1):
        self.num_data_used = num_data_used
        self.bgfg = bgfg
        self.num_class = 10
        super(FileIter, self).__init__()
        self.file_lst = file_lst
        f_class = open(class_names, 'r')
        count = 1
        self.class_names = {}
        self.class_names[0] = "background"
        for line in f_class:
            self.class_names[line.strip('\n')] = count
            self.class_names[count] = line.strip('\n')
            count += 1
        self.rgb_mean = np.array(rgb_mean)
        f_class.close()

        self.mean_face = np.zeros([self.num_class, 3], np.float32)
        f_mface = open(file_mface, 'r')
        for line in f_mface:
            line = line.strip('\n').split(' ')
            xyz = np.array([float(line[1]), float(line[2]), float(line[3])])
            self.mean_face[self.class_names[line[0]] - 1] = xyz
        f_mface.close()
        self.AllImg = {}

        self.data_name = data_name
        self.cls_label_name = cls_label_name
        self.proj_label_name = proj_label_name
        self.proj_weight_name = proj_weight_name
        self.ground_truth_name = ground_truth_name
        self.bbox_label_name = bbox_label_name
        self.bbox_weight_name = bbox_weight_name
        self.num_data = len(open(self.file_lst, 'r').readlines())
        self.f = open(self.file_lst, 'r')
        self.cursor = -1
        self.data, self.cls_label, self.proj_label, self.proj_weight, self.ground_truth,\
            self.bbox_label, self.bbox_weight = [], [], [], [], [], [], []

    #####################

    def _read(self):
        self.AllImg[self.cursor] = myImage()
        line = self.f.readline()
        onedata = line.strip('\n').split(' ')
        imgFilename = onedata[1]

        self.AllImg[self.cursor].filename = imgFilename
        img_width, img_height, resize_factor = [float(item) for item in onedata[2:5]]
        wd_resize = int(img_width * resize_factor)
        ht_resize = int(img_height * resize_factor)
        self.AllImg[self.cursor].width = wd_resize
        self.AllImg[self.cursor].height = ht_resize
        self.AllImg[self.cursor].resize_factor = resize_factor
        img = Image.open(imgFilename).convert("RGB")
        if_blur = random.randint(1, 4)
        blur_radius = random.randint(5, 20)
        if if_blur == 1:
            print 'Blured: ', blur_radius
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = img.resize((wd_resize, ht_resize))
        img = np.array(img, dtype=np.float32)

        self.AllImg[self.cursor].image = []
        num_faces = int(onedata[5])
        num_keypoints = 0
        self.AllImg[self.cursor].num_faces = num_faces
        faces = {}
        onedata = onedata[6:]
        for j in range(0, num_faces):
            faces[j], num_k, onedata = self._read_oneface(onedata)
            num_keypoints += num_k
        self.AllImg[self.cursor].faces = faces

        tempt_model = get_vgg16_rpn()
        out_height, out_width = tempt_model.infer_shape(data=(1, 3, ht_resize, wd_resize), mean_face=(10, 3),
                                                        ground_truth=(10, 2), bbox_label=(10, 5))[1][0][2:4]
        cls_label = 255 * np.ones((out_height, out_width), dtype=np.int32)
        proj_label = np.zeros((out_height, out_width, self.num_class, 2), dtype=np.int32)
        proj_weight = np.zeros((out_height, out_width, self.num_class, 2), dtype=np.float32)
        ground_truth = np.zeros((num_keypoints * 9, 2), dtype=np.int32)
        bbox_label = np.zeros((num_keypoints * 9, 5), dtype=np.float32)
        bbox_weight = np.zeros((num_keypoints * 9, 5), dtype=np.float32)

        ratio_w, ratio_h = float(out_width) / float(wd_resize), float(out_height) / float(ht_resize)
        count = 0

        flag_map = np.ones((out_height, out_width))  # used to record negative anchors selection

        gt_counter = 0
        for i in range(0, num_faces):
            kp_arr = np.zeros((self.num_class, 2), dtype=np.int32)
            kp_warr = np.zeros((self.num_class, 2), dtype=np.float32)
            for kname, kp in faces[i].keypoints.iteritems():
                kp_arr[self.class_names[kname] - 1] = np.array([kp.y, kp.x])
                kp_warr[self.class_names[kname] - 1] = \
                    np.array([100.0 / faces[i].bbox.height, 100.0 / faces[i].bbox.height])
            for kname, kp in faces[i].keypoints.iteritems():
                for idx_y in range(-1, 2):
                    if (kp.y + idx_y) * ratio_h >= 0 and (kp.y + idx_y) * ratio_h < out_height:
                        for idx_x in range(-1, 2):
                            if (kp.x + idx_x) * ratio_w >= 0 and (kp.x + idx_x) * ratio_w < out_width:
                                ground_truth[gt_counter, :] = np.array([kp.y + idx_y, kp.x + idx_x])
                                bbox_label[gt_counter, :] = np.array([faces[i].ellipse[0],
                                                                      faces[i].ellipse[1],
                                                                      faces[i].ellipse[3],
                                                                      faces[i].ellipse[4],
                                                                      faces[i].ellipse[2]])
                                bbox_weight[gt_counter, :] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
                                gt_counter += 1

            for kname, kp in faces[i].keypoints.iteritems():
                pos_x, pos_y = int(kp.x * ratio_w), int(kp.y * ratio_h)

                if pos_x > out_width or pos_y > out_height:
                    print self.AllImg[self.cursor].filename
                    print pos_x, pos_y, out_width, out_height, wd_resize, ht_resize, ratio_w, ratio_h
                count += 1
                for idx_y in range(-1, 2):
                    if pos_y + idx_y >= 0 and pos_y + idx_y < out_height:
                        for idx_x in range(-1, 2):
                            if pos_x + idx_x >= 0 and pos_x + idx_x < out_width:
                                if self.bgfg:
                                    cls_label[pos_y + idx_y, pos_x + idx_x] = 1
                                else:
                                    cls_label[pos_y + idx_y, pos_x + idx_x] = self.class_names[kname]
                                proj_label[pos_y + idx_y, pos_x + idx_x] = kp_arr
                                proj_weight[pos_y + idx_y, pos_x + idx_x] = kp_warr

        for i in range(0, num_faces):
            bbox = faces[i].bbox
            y = int(max(0, math.floor(bbox.y * ratio_h)))
            ty = int(min(out_height - 1, math.floor((bbox.y + bbox.height) * ratio_h)))
            x = int(max(0, math.floor(bbox.x * ratio_w)))
            tx = int(min(out_width - 1, math.floor((bbox.x + bbox.width) * ratio_w)))
            flag_map[y: ty, x: tx] = np.zeros((ty - y, tx - x))

        # random choose negative anchors
        for i in range(0, 9 * count):
            left_anchor = np.nonzero(flag_map)
            if left_anchor[0].size:
                index_neg = random.randint(0, left_anchor[0].size - 1)
                cls_label[left_anchor[0][index_neg], left_anchor[1][index_neg]] = 0
                flag_map[left_anchor[0][index_neg], left_anchor[1][index_neg]] = 0
            else:
                break

        img = img - self.rgb_mean
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c,h,w)
        img = np.expand_dims(img, axis=0)  # (1,c,h,w) 1 is batch number

        cls_label = np.expand_dims(cls_label, axis=0)
        proj_label = np.expand_dims(proj_label, axis=0)
        proj_weight = np.expand_dims(proj_weight, axis=0)

        return img, cls_label, proj_label, proj_weight, ground_truth[:gt_counter], \
               bbox_label[:gt_counter], bbox_weight[:gt_counter]

    def _read_oneface(self, str):
        oneface = Face()
        # every face has bouding box and proj bounding box
        oneface.bbox.x, oneface.bbox.y, oneface.bbox.width, oneface.bbox.height, \
        oneface.proj_bbox.x, oneface.proj_bbox.y, oneface.proj_bbox.width, oneface.proj_bbox.height = [
            int(float(item)) for item in str[0:8]]

        num_keypoint = int(str[8])
        oneface.num_keypoint = num_keypoint

        # every keypoint has name, x, y proj_x, proj_y
        for i in range(0, num_keypoint):
            info_kp = str[9 + i * 5:9 + (i + 1) * 5]
            name_kp = info_kp[0]
            oneface.keypoints[name_kp] = feature_pos(info_kp[1:5])

        ellipse = str[9 + num_keypoint * 5: 14 + num_keypoint * 5]
        oneface.ellipse = np.array([float(item) for item in ellipse])
        if oneface.ellipse[2] < 0:
            oneface.ellipse[2] += np.pi
        return oneface, num_keypoint, str[26 + num_keypoint * 5:]

    def get_batch_size(self):
        return 1

    def reset(self):
        self.cursor = -1
        self.f.close()
        self.f = open(self.file_lst, 'r')

    def iter_next(self):
        self.cursor += 1
        if self.num_data_used != -1 and self.cursor > self.num_data_used:
            return False
        if (self.cursor < self.num_data):
            return True
        else:
            return False

    def next(self):
        # return a dictionary contains all data needed for one iteration
        if self.iter_next():
            self.data, self.cls_label, self.proj_label, self.proj_weight, self.ground_truth,\
                self.bbox_label, self.bbox_weight = self._read()
            return {self.data_name: self.data,
                    self.cls_label_name: self.cls_label,
                    self.proj_label_name: self.proj_label,
                    self.proj_weight_name: self.proj_weight,
                    self.ground_truth_name: self.ground_truth,
                    self.bbox_label_name: self.bbox_label,
                    self.bbox_weight_name: self.bbox_weight}
        else:
            raise StopIteration
