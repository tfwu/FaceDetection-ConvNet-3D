import numpy as np
from PIL import Image
import cv2
import mxnet as mx
import fddb_symbol_vgg16
import fddb_symbol_gen
import fddb_symbol_finetune
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

num_cls = 10
channel_len = 64
feature_len = 3136 + 8 + num_cls * (num_cls + 1) + num_cls * channel_len
label_len = 6

rpn_prefix = "model_vgg16/VGG16"
finetune_prefix = "model_finetune/finetune"
rpn_epoch = 813001
finetune_epoch = 201

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(2)
rgb_mean = np.array([123.68, 116.779, 103.939])
file_class = "file/class.txt"
file_mface = "file/model3d.txt"

file_lst = "file/test.txt"

MAX_FACE_NUM = 50
num_proposal = 300
nms_iou = 0.5
nms_ios = 0.6

iou_self_threshold = 0.8

keep_ratio = 0.6666667

metric_ellipse = False


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


def convert_ell2rec_afw(ell, keep_ratio):
    major_height = ell[0] * np.sin(ell[2])
    major_width = ell[0] * np.cos(ell[2])
    minor_height = ell[1] * np.sin(np.pi / 2.0 - ell[2])
    minor_width = ell[1] * np.cos(np.pi / 2.0 - ell[2])
    center_between_eyebrow = np.array([ell[3] - major_height * (keep_ratio - 0.5),
                                       ell[4] - major_width * (keep_ratio - 0.5)])
    chin = np.array([ell[3] + major_height * 0.5, ell[4] + major_width * 0.5])
    left_border = np.array([ell[3] + minor_height * 0.5, ell[4] - minor_width * 0.5])
    right_border = np.array([ell[3] - minor_height * 0.5, ell[4] + minor_width * 0.5])

    return [min(center_between_eyebrow[0], chin[0], left_border[0], right_border[0]),
            min(center_between_eyebrow[1], chin[1], left_border[1], right_border[1]),
            max(center_between_eyebrow[0], chin[0], left_border[0], right_border[0]),
            max(center_between_eyebrow[1], chin[1], left_border[1], right_border[1])]


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


def non_maximum_suppression_rpn(ground_truth, faceness, box_predict, spatial_scale):
    num_gt = ground_truth.shape[0]
    ret_gt = np.zeros((num_gt, 2), dtype=np.float32)
    ret_faceness = np.zeros(num_gt, dtype=np.float32)
    num_gt_ret = 0

    rec_ret = np.zeros((num_gt, 4), dtype=np.float32)

    for i in xrange(num_gt):
        if faceness[i] < -100:
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
            ret_faceness[num_gt_ret] = faceness[i]
            rec_ret[num_gt_ret] = np.array(box_predict[0,
                                           int(ret_gt[num_gt_ret, 0] * spatial_scale),
                                           int(ret_gt[num_gt_ret, 1] * spatial_scale), :])
            rec_ret[num_gt_ret, 2] += rec_ret[num_gt_ret, 0]
            rec_ret[num_gt_ret, 3] += rec_ret[num_gt_ret, 1]
            num_gt_ret += 1

    return ret_gt[:num_gt_ret], ret_faceness[:num_gt_ret]


def calc_ellipse(ell_output, predict_bbox_output, ground_truth, spatial_scale):
    ellipses = np.zeros((ell_output.shape[0], 5), dtype=np.float)
    for i in xrange(ell_output.shape[0]):
        predict_bbox = predict_bbox_output[0,
                       int(ground_truth[i, 0] * spatial_scale),
                       int(ground_truth[i, 1] * spatial_scale), :]
        ellipses[i] = np.array([ell_output[i, 0] * predict_bbox[2] * 2,
                                ell_output[i, 1] * predict_bbox[3] * 2,
                                ell_output[i, 4],
                                ell_output[i, 2] * predict_bbox[2] + predict_bbox[0],
                                ell_output[i, 3] * predict_bbox[3] + predict_bbox[1]])

    return ellipses


def calc_ios_rectangle(rec0, rec1):
    tempt = np.array([rec0, rec1])
    l, u = tempt.max(axis=0)[0:2]
    r, d = tempt.min(axis=0)[2:4]
    if l >= r or u >= d:
        return 0

    insec = (r - l) * (d - u)
    area1 = (rec0[2] - rec0[0]) * (rec0[3] - rec0[1])
    area2 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    if area1 == 0 or area2 == 0:
        return 0
    return max(float(insec) / float(area1), float(insec) / float(area2))


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


def calc_iou_ellipse(ell0, ell1):
    return calc_iou_rectangle(convert_ell2rec(ell0), convert_ell2rec(ell1))


def calc_ios_ellipse(ell0, ell1):
    return calc_ios_rectangle(convert_ell2rec(ell0), convert_ell2rec(ell1))


def non_maximum_suppression(ellipses, faceness, predict_iou, iou, ios):
    res_ellipse = np.zeros((num_proposal, 5))
    res_faceness = np.zeros(num_proposal)

    num_ellipse = ellipses.shape[0]
    num_res = 0

    for i in xrange(num_ellipse):

        if predict_iou[i] < 0.1:
            continue

        flag = True
        for j in xrange(num_res):
            if calc_iou_ellipse(res_ellipse[j], ellipses[i]) >= iou or \
                            calc_ios_ellipse(res_ellipse[j], ellipses[i]) >= ios:
                flag = False
                break

        if flag:
            res_ellipse[num_res] = np.array(ellipses[i])
            res_faceness[num_res] = np.array((faceness[i]))
            num_res += 1
            if num_res == num_proposal:
                break

    return res_ellipse[:num_res], res_faceness[:num_res], num_res


def predict():
    # Record the name of the classes
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

    # Load model
    vgg16_rpn = fddb_symbol_vgg16.get_vgg16_rpn()
    _, rpn_args, rpn_auxs = mx.model.load_checkpoint(rpn_prefix, rpn_epoch)
    rpn_args, rpn_auxs = init_from_vgg16(ctx, vgg16_rpn, rpn_args, rpn_auxs)

    vgg16_gen = fddb_symbol_gen.get_vgg16_gen()
    gen_args = {}
    gen_auxs = {}

    vgg16_finetune = fddb_symbol_finetune.get_vgg16_finetune()
    _, finetune_args, finetune_auxs = mx.model.load_checkpoint(finetune_prefix + '-09', finetune_epoch)
    finetune_args, finetune_auxs = init_from_vgg16(ctx, vgg16_finetune, finetune_args, finetune_auxs)

    rpn_args["mean_face"] = mx.nd.array(mean_face, ctx)

    f = open(file_lst, 'r')
    for line in f:
        line = line.strip('\n').split(' ')
        imgFilename = line[0]
        img = Image.open(imgFilename)
        width, height = img.size
        short_len = min(width, height)
        resize_factor = 600.0 / float(short_len)
        long_len = max(width * resize_factor, height * resize_factor)
        if long_len > 1000:
            resize_factor = 1000.0 / float(max(width, height))

        width = int(resize_factor * width)
        height = int(resize_factor * height)

        print "image shape", height, width

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

        ground_truth, tmp_faceness = non_maximum_suppression_rpn(ground_truth, tmp_faceness, box_predict_output,
                                                                 spatial_scale)

        print "rpn prediction shape:", ground_truth.shape

        f_rpn = open('file/rpn_bbox.txt', 'w')
        for i in xrange(ground_truth.shape[0]):
            px = int(ground_truth[i][0] * spatial_scale)
            py = int(ground_truth[i][1] * spatial_scale)
            sx = box_predict_output[0, px, py, 0]
            sy = box_predict_output[0, px, py, 1]
            ex = box_predict_output[0, px, py, 2]
            ey = box_predict_output[0, px, py, 3]
            f_rpn.write(str(sy) + " " + str(sx) + " " + str(ey) + " " + str(ex) + "\n")
        f_rpn.close()

        #################
        ### vgg16 gen ###
        #################

        gen_args["ground_truth"] = mx.nd.array(ground_truth, ctx)
        gen_args["box_predict"] = mx.nd.array(box_predict_output, ctx)
        gen_args["relu_feature"] = mx.nd.array(relu_feature_output, ctx)
        gen_args["bbox_label"] = mx.nd.array(np.zeros((ground_truth.shape[0], label_len)), ctx)
        executor = vgg16_gen.bind(ctx, gen_args, args_grad=None, grad_req="null", aux_states=gen_auxs)
        executor.forward(is_train=True)
        roi_flatten_output = mx.nd.zeros(executor.outputs[0].shape)
        ell_label_output = mx.nd.zeros(executor.outputs[1].shape)
        executor.outputs[0].copyto(roi_flatten_output)
        executor.outputs[1].copyto(ell_label_output)
        roi_flatten_output = roi_flatten_output.asnumpy()
        ell_label_output = ell_label_output.asnumpy()

        num_sample = roi_flatten_output.shape[0]
        feature_length = roi_flatten_output.shape[1]
        feature_b = np.zeros((num_sample, feature_len), dtype=np.float)
        for i in xrange(num_sample):
            index_feature = 0
            for j in xrange(feature_length):
                feature_b[i, index_feature] = roi_flatten_output[i, j]
                index_feature += 1

            px = int(ground_truth[i, 0] * spatial_scale)
            py = int(ground_truth[i, 1] * spatial_scale)

            for j in xrange(num_class - 1):
                pxx = int(face3dproj_output[0, px, py, j * 2] * spatial_scale)
                pyy = int(face3dproj_output[0, px, py, j * 2 + 1] * spatial_scale)
                for k in xrange(channel_len):
                    if pxx >= 0 and pxx < out_height and pyy >= 0 and pyy < out_width:
                        feature_b[i, index_feature] = relu_feature_output[0, k, pxx, pyy]
                        index_feature += 1
                    else:
                        feature_b[i, index_feature] = 0.0
                        index_feature += 1

            for j in xrange(num_class - 1):
                pxx = int(face3dproj_output[0, px, py, j * 2] * spatial_scale)
                pyy = int(face3dproj_output[0, px, py, j * 2 + 1] * spatial_scale)
                for k in xrange(num_class):
                    if pxx >= 0 and pxx < out_height and pyy >= 0 and pyy < out_width:
                        feature_b[i, index_feature] = softmax_output[k, pxx, pyy]
                        index_feature += 1
                    else:
                        feature_b[i, index_feature] = 0.0
                        index_feature += 1

            for j in xrange(8):
                feature_b[i, index_feature] = param3d_pred_output[0, j, px, py]
                index_feature += 1

        mean = np.load('./mean.npy')
        var = np.load('./var.npy')
        feature_b = (feature_b - mean) / var

        ######################
        ### vgg16 finetune ###
        ######################

        finetune_args["data"] = mx.nd.array(feature_b, ctx)
        finetune_args["ell_label"] = mx.nd.array(ell_label_output, ctx)
        finetune_args["bbox_weight"] = mx.nd.array(np.ones((num_sample, label_len), dtype=np.float), ctx)
        executor = vgg16_finetune.bind(ctx, finetune_args, args_grad=None, grad_req="null", aux_states=finetune_auxs)
        executor.forward(is_train=True)
        ell_output = mx.nd.zeros(executor.outputs[0].shape)
        executor.outputs[0].copyto(ell_output)
        ell_output = ell_output.asnumpy()

        tmp_ellipses = calc_ellipse(ell_output, box_predict_output, ground_truth, spatial_scale)

        print "final prediction shape:", tmp_ellipses.shape

        f = open("file/final_ell_before_nms.txt", "w")
        for i in xrange(tmp_ellipses.shape[0]):
            f.write(str(tmp_ellipses[i, 0]) + " " + str(tmp_ellipses[i, 1]) + " " + str(tmp_ellipses[i, 2]) + " " +
                    str(tmp_ellipses[i, 3]) + " " + str(tmp_ellipses[i, 4]) + " " + str(ell_output[i, 5]) + "\n")
        f.close()

        ellipses, faceness, num_ellipses = \
            non_maximum_suppression(tmp_ellipses, tmp_faceness, ell_output[:, 5], nms_iou, nms_ios)

        print "final prediction shape:", ellipses.shape

        f = open("file/final_ell.txt", "w")
        for i in xrange(num_ellipses):
            f.write(str(ellipses[i, 0]) + " " + str(ellipses[i, 1]) + " " + str(ellipses[i, 2]) + " " +
                    str(ellipses[i, 3]) + " " + str(ellipses[i, 4]) + "\n")
        f.close()

        f = open("file/final_bbox.txt", 'w')
        for i in xrange(num_ellipses):
            rect = convert_ell2rec_afw(ellipses[i], keep_ratio)
            f.write(str(rect[1]) + " " + str(rect[0]) + " " +
                    str(rect[3] - rect[1]) + " " + str(rect[2] - rect[0]) + " " +
                    str(faceness[i]) + "\n")

        f.close()


if __name__ == "__main__":
    predict()
