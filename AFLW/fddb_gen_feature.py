import numpy as np
from PIL import Image
import cv2
import mxnet as mx
import fddb_symbol_vgg16
import fddb_symbol_gen
from fddb_init_vgg16 import init_from_vgg16
import logging

from PIL import Image
import numpy as np

# 0 LeftEyeLeftCorner
# 1 RightEyeRightCorner
# 2 LeftEar
# 3 NoseLeft
# 4 NoseRight
# 5 RightEar
# 6 MouthLeftCorner
# 7 MouthRightCorner
# 8 ChinCenter
# 9 center_between_eyes

INF = 0x3f3f3f3f
spatial_scale = 0.5
num_class = 11
num_channel = 64

rpn_prefix = "model_vgg16/VGG16"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(2)
epoch = 813001
rgb_mean = np.array([123.68, 116.779, 103.939])
file_class = "file/class.txt"
file_mface = "file/model3d.txt"

path_dataset = "/home/yunzhu/face/FDDB/"
img_format = ".jpg"

MAX_FACE_NUM = 30
num_proposal = 300
num_sample_per_face = 100

iou_self_threshold = 0.8
iou_gt_threshold = 0.3
select_count = 500

def calc_length(vec):
    return np.sqrt(np.sum(vec ** 2))


def convert_ell2rec(ell):
    point = np.zeros((4, 2))

    height = np.abs(ell[0] * np.sin(ell[2]))
    width = np.abs(ell[0] * np.cos(ell[2]))
    point[0] = np.array([ell[3] - height / 2.0, ell[4] - width / 2.0])
    point[1] = np.array([ell[3] + height / 2.0, ell[4] + width / 2.0])

    height = np.abs(ell[1] * np.sin(ell[2] - np.pi / 2.0))
    width = np.abs(ell[1] * np.cos(ell[2] - np.pi / 2.0))
    point[2] = np.array([ell[3] - height / 2.0, ell[4] - width / 2.0])
    point[3] = np.array([ell[3] + height / 2.0, ell[4] + width / 2.0])

    return [min(point[0][0], point[2][0]), min(point[0][1], point[2][1]),
            max(point[1][0], point[3][0]), max(point[1][1], point[3][1])]


def calc_iou_rectangle(rec0, rec1):
    tempt = np.array([rec0, rec1])
    l, u = tempt.max(axis=0)[0:2]
    r, d = tempt.min(axis=0)[2:4]
    if l >= r or u >= d:
        return 0

    insec = (r - l) * (d - u)
    area = (rec0[2] - rec0[0]) * (rec0[3] - rec0[1]) + \
           (rec1[2] - rec1[0]) * (rec1[3] - rec1[1]) - insec
    if area == 0:
        return 0
    return float(insec) / float(area)


def calc_ground_truth(height, width, num_cls, tensor_keypoint, tensor_softmax, argmax_label):
    tensor_keypoint = tensor_keypoint.reshape((height, width, num_cls - 1, 2))
    ground_truth = np.zeros((height * width, 2))
    faceness = np.zeros((height * width))
    num_ground_truth = 0

    for i in xrange(height):
        for j in xrange(width):
            if argmax_label[i, j] != 0:
                ground_truth[num_ground_truth] = np.array([i * 2, j * 2])
                cnt_inside_img = 0
                for k in xrange(num_cls - 1):
                    predict_x = int(tensor_keypoint[i, j, k, 0] * spatial_scale)
                    predict_y = int(tensor_keypoint[i, j, k, 1] * spatial_scale)
                    if predict_x >= 0 and predict_x < height and predict_y >= 0 and predict_y < width:
                        faceness[num_ground_truth] += np.log(tensor_softmax[k + 1, predict_x, predict_y])
                        cnt_inside_img += 1

                if cnt_inside_img < 3:
                    faceness[num_ground_truth] = -INF
                else:
                    faceness[num_ground_truth] = faceness[num_ground_truth] / cnt_inside_img * (num_cls - 1)

                num_ground_truth += 1

    return ground_truth[:num_ground_truth], faceness[:num_ground_truth]


def inside_rec(p, rec):
    return p[0] >= rec[0] and p[0] <= rec[2] and p[1] >= rec[1] and p[1] <= rec[3]


def nms_and_calc_label(ground_truth, faceness, box_predict, ellipses_gt, spatial_scale):
    num_ell = ellipses_gt.shape[0]
    num_gt = ground_truth.shape[0]
    ret_gt = np.zeros((num_gt, 2), dtype=np.float32)
    ret_label = np.zeros((num_gt, 5), dtype=np.float32)
    ret_iou = np.zeros(num_gt, dtype=np.float32)
    num_gt_ret = 0

    rec_ret = np.zeros((num_gt, 4), dtype=np.float32)

    num_greater = 0
    num_smaller = 0

    for i in xrange(num_gt):
        if faceness[i] < -80:
            break

        flag = True
        rec0 = np.array(box_predict[0,
                        int(ground_truth[i, 0] * spatial_scale),
                        int(ground_truth[i, 1] * spatial_scale), :])
        rec0[2] += rec0[0]
        rec0[3] += rec0[1]

        for j in xrange(num_gt_ret):
            if calc_iou_rectangle(rec0, rec_ret[j]) > iou_self_threshold:
                flag = False
                break

        if flag:
            ret_gt[num_gt_ret] = ground_truth[i]
            iou = 0
            for j in xrange(num_ell):
                rec1 = convert_ell2rec(ellipses_gt[j])
                if calc_iou_rectangle(rec0, rec1) > iou:
                    iou = calc_iou_rectangle(rec0, rec1)
                    ret_label[num_gt_ret, :] = np.array([ellipses_gt[j, 0] / 2.0,
                                                         ellipses_gt[j, 1] / 2.0,
                                                         ellipses_gt[j, 3],
                                                         ellipses_gt[j, 4],
                                                         ellipses_gt[j, 2]])

            ret_iou[num_gt_ret] = iou

            rec_ret[num_gt_ret] = np.array(box_predict[0,
                                            int(ret_gt[num_gt_ret, 0] * spatial_scale),
                                            int(ret_gt[num_gt_ret, 1] * spatial_scale), :])
            rec_ret[num_gt_ret, 2] += rec_ret[num_gt_ret, 0]
            rec_ret[num_gt_ret, 3] += rec_ret[num_gt_ret, 1]
            num_gt_ret += 1

            if iou >= iou_gt_threshold:
                num_greater += 1
            else:
                num_smaller += 1

    print "num > 0.3:", num_greater, "num < 0.3:", num_smaller

    return ret_gt[:num_gt_ret], ret_label[:num_gt_ret], ret_iou[:num_gt_ret]


def gen_fddb_feature(fold_index):
    file_fold = "/home/yunzhu/face_ext/FDDB/FDDB-folds/FDDB-fold-" + fold_index + "-ellipseList.txt"
    file_lst = "/home/yunzhu/face_ext/FDDB/FDDB-folds/FDDB-fold-" + fold_index + ".txt"
    file_feature = "/home/yunzhu/face_ext/mxnet/AFLW/fddb_feature/feature-" + fold_index + ".txt"

    f_feature = open(file_feature, "w")

    f_class = open(file_class, 'r')
    count = 1
    class_names = {}
    class_names[0] = "background"
    for line in f_class:
        class_names[line.strip('\n')] = count
        class_names[count] = line.strip('\n')
        count += 1
    f_class.close()

    # Record the mean face
    mean_face = np.zeros([num_class - 1, 3], np.float32)
    f_mface = open(file_mface, 'r')
    for line in f_mface:
        line = line.strip('\n').split(' ')
        xyz = np.array([float(line[1]), float(line[2]), float(line[3])])
        mean_face[class_names[line[0]] - 1] = xyz
    f_mface.close()

    vgg16_rpn = fddb_symbol_vgg16.get_vgg16_rpn()
    _, rpn_args, rpn_auxs = mx.model.load_checkpoint(rpn_prefix, epoch)
    rpn_args, rpn_auxs = init_from_vgg16(ctx, vgg16_rpn, rpn_args, rpn_auxs)

    rpn_args["mean_face"] = mx.nd.array(mean_face, ctx)

    vgg16_gen = fddb_symbol_gen.get_vgg16_gen()
    _, gen_args, gen_auxs = mx.model.load_checkpoint(rpn_prefix, epoch)
    gen_args, gen_auxs = init_from_vgg16(ctx, vgg16_gen, gen_args, gen_auxs)

    f = open(file_lst, 'r').readlines()
    num_img = len(f)

    ellipses_gt = np.zeros((num_img, MAX_FACE_NUM, 5))
    num_face_gt = np.zeros((num_img,), dtype=int)

    f = open(file_fold, 'r').readlines()
    index_upp = len(f)
    index = 0

    index_img = 0

    while index < index_upp:

        print "Image index: [" + fold_index + "] %d/%d" % (index_img, num_img)

        file_img = path_dataset + f[index].strip() + img_format
        img = Image.open(file_img)
        width, height = img.size
        short_len = min(width, height)
        resize_factor = 600.0 / float(short_len)
        long_len = max(width * resize_factor, height * resize_factor)
        if long_len > 1000:
            resize_factor = 1000.0 / float(max(width, height))

        index += 1
        num_face_gt[index_img] = int(f[index].strip())
        print "num face:", num_face_gt[index_img]
        for i in xrange(num_face_gt[index_img]):
            index += 1
            major, minor, angle, y, x, tmp = [float(x) for x in f[index].strip().split()]
            if angle < 0:
                angle += np.pi
            ellipses_gt[index_img][i] = np.array([major * resize_factor * 2,
                                                  minor * resize_factor * 2,
                                                  angle,
                                                  x * resize_factor,
                                                  y * resize_factor])

        width = int(resize_factor * width)
        height = int(resize_factor * height)

        out_height, out_width = vgg16_rpn.infer_shape(data=(1, 3, height, width), mean_face=(10, 3))[1][0][2:4]

        cls_label = mx.nd.empty((1, out_height * out_width), ctx)
        proj_label = mx.nd.empty((1, out_height, out_width, 10, 2), ctx)
        proj_weight = mx.nd.empty((1, out_height, out_width, 10, 2), ctx)
        rpn_args["cls_label"] = cls_label
        rpn_args["proj_label"] = proj_label
        rpn_args["proj_weight"] = proj_weight
        img = img.resize((int(width), int(height)))
        img = np.array(img, dtype=np.float32)

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = img - rgb_mean
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = np.expand_dims(img, axis=0)

        rpn_args["data"] = mx.nd.array(img, ctx)
        executor = vgg16_rpn.bind(ctx, rpn_args, args_grad=None, grad_req="null", aux_states=rpn_auxs)
        executor.forward(is_train=True)

        softmax_output = mx.nd.zeros(executor.outputs[0].shape)
        regression_output = mx.nd.zeros(executor.outputs[1].shape)
        relu_feature_output = mx.nd.zeros(executor.outputs[2].shape)
        face3dproj_output = mx.nd.zeros(executor.outputs[3].shape)
        box_predict_output = mx.nd.zeros(executor.outputs[4].shape)
        param3d_pred_output = mx.nd.zeros(executor.outputs[5].shape)
        executor.outputs[0].copyto(softmax_output)
        executor.outputs[1].copyto(regression_output)
        executor.outputs[2].copyto(relu_feature_output)
        executor.outputs[3].copyto(face3dproj_output)
        executor.outputs[4].copyto(box_predict_output)
        executor.outputs[5].copyto(param3d_pred_output)
        softmax_output = np.squeeze(softmax_output.asnumpy())
        regression_output = np.squeeze(regression_output.asnumpy())
        argmax_label = np.uint8(softmax_output.argmax(axis=0))

        relu_feature_output = relu_feature_output.asnumpy()
        face3dproj_output = face3dproj_output.asnumpy()
        box_predict_output = box_predict_output.asnumpy()
        param3d_pred_output = param3d_pred_output.asnumpy()

        ground_truth, tmp_faceness = calc_ground_truth(out_height, out_width, num_class, regression_output,
                                                       softmax_output, argmax_label)

        index_lst = np.argsort(-tmp_faceness)
        ground_truth = ground_truth[index_lst, :]
        tmp_faceness = tmp_faceness[index_lst]

        print ground_truth.shape

        # if ground_truth.shape[0] > 40000:
        #     ground_truth = np.array([ground_truth[k, :] for k in xrange(0, ground_truth.shape[0], 2)])

        ground_truth, label, iou = nms_and_calc_label(ground_truth, tmp_faceness, box_predict_output,
                                                      ellipses_gt[index_img, :num_face_gt[index_img]], spatial_scale)

        print ground_truth.shape

        if ground_truth.shape[0] == 0:
            index += 1
            index_img += 1
            continue

        gen_args["ground_truth"] = mx.nd.array(ground_truth, ctx)
        gen_args["box_predict"] = mx.nd.array(box_predict_output, ctx)
        gen_args["relu_feature"] = mx.nd.array(relu_feature_output, ctx)
        gen_args["bbox_label"] = mx.nd.array(label, ctx)
        executor = vgg16_gen.bind(ctx, gen_args, args_grad=None, grad_req="null", aux_states=gen_auxs)
        executor.forward(is_train=True)
        roi_flatten_output = mx.nd.zeros(executor.outputs[0].shape)
        ell_label_output = mx.nd.zeros(executor.outputs[1].shape)
        executor.outputs[0].copyto(roi_flatten_output)
        executor.outputs[1].copyto(ell_label_output)
        roi_flatten_output = roi_flatten_output.asnumpy()
        ell_label_output = ell_label_output.asnumpy()

        print "roi_flatten_output shape:", roi_flatten_output.shape
        print "ell_label_output shape:", ell_label_output.shape

        num_sample = roi_flatten_output.shape[0]
        feature_length = roi_flatten_output.shape[1]
        for i in xrange(num_sample):

            for j in xrange(feature_length):
                f_feature.write("%f " % (roi_flatten_output[i, j]))

            px = int(ground_truth[i, 0] * spatial_scale)
            py = int(ground_truth[i, 1] * spatial_scale)

            for j in xrange(num_class - 1):
                pxx = int(face3dproj_output[0, px, py, j * 2] * spatial_scale)
                pyy = int(face3dproj_output[0, px, py, j * 2 + 1] * spatial_scale)
                for k in xrange(num_channel):
                    if pxx >= 0 and pxx < out_height and pyy >= 0 and pyy < out_width:
                        f_feature.write("%f " % relu_feature_output[0, k, pxx, pyy])
                    else:
                        f_feature.write("0.0 ")

            for j in xrange(num_class - 1):
                pxx = int(face3dproj_output[0, px, py, j * 2] * spatial_scale)
                pyy = int(face3dproj_output[0, px, py, j * 2 + 1] * spatial_scale)
                for k in xrange(num_class):
                    if pxx >= 0 and pxx < out_height and pyy >= 0 and pyy < out_width:
                        f_feature.write("%f " % softmax_output[k, pxx, pyy])
                    else:
                        f_feature.write("0.0 ")

            for j in xrange(8):
                f_feature.write("%f " % param3d_pred_output[0, j, px, py])

            f_feature.write("\n")

            for j in xrange(5):
                f_feature.write("%f " % (ell_label_output[i, j]))

            f_feature.write("\n")

            f_feature.write("%f\n" % (iou[i]))

        index += 1
        index_img += 1

    f_feature.close()


target_index = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
# target_index = ["08", "09", "10"]

if __name__ == "__main__":
    for i in target_index:
        gen_fddb_feature(i)
