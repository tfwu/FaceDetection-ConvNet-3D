import numpy as np
from PIL import Image
import cv2
import mxnet as mx
import logging
import fddb_symbol_vgg16
import fddb_symbol_fc
from fddb_init_vgg16 import init_from_vgg16

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
num_classes = 11
num_proposal = 300
nms_iou = 0.5
nms_ios = 0.6
metric_iou = 0.5


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


def calc_ground_truth(height, width, num_cls, tensor_keypoint, tensor_softmax, argmax_label):
    tensor_keypoint = tensor_keypoint.reshape((height, width, num_cls - 1, 2))
    ground_truth = np.zeros((height * width, 2), dtype=np.int32)
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


def calc_ellipse(ell_output, predict_bbox_output):
    ellipses = np.zeros(ell_output.shape)
    for i in xrange(ell_output.shape[0]):
        ellipses[i] = np.array([ell_output[i, 0] * predict_bbox_output[i, 2] * 2,
                                ell_output[i, 1] * predict_bbox_output[i, 3] * 2,
                                ell_output[i, 4] - np.pi,
                                ell_output[i, 2] * predict_bbox_output[i, 2] + predict_bbox_output[i, 0],
                                ell_output[i, 3] * predict_bbox_output[i, 3] + predict_bbox_output[i, 1]])

    return ellipses


def non_maximum_suppression(ellipses, faceness, iou, ios):
    res_ellipse = np.zeros((num_proposal, 5))
    res_faceness = np.zeros(num_proposal)

    num_ellipse = ellipses.shape[0]
    num_res = 0

    for i in xrange(num_ellipse):

        if faceness[i] < -100.0:
            break

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

    return res_ellipse, res_faceness, num_res

pallete = [0, 0, 0,
           128, 0, 0,  # LeftEyeLeftCorner   dark red
           0, 128, 0,  # RightEyeRightCorner   dark green
           128, 128, 0,  # LeftEar   dark yellow
           0, 0, 128,  # NoseLeft    dark blue
           128, 0, 128,  # NoseRight    dark purple
           0, 128, 128,  # RightEar    qingse
           128, 128, 128,  # MouthLeftCorner   gray
           64, 0, 0,  # MouthRightCorner     darker red
           255, 0, 0,  # ChinCenter    red
           0, 0, 255,  # center_between_eyes   blue
           ]

rpn_prefix = "model_vgg16/VGG16"
rpn_epoch = 813001
fc_prefix = "model_finetune/finetune-01"
fc_epoch = 76
is_finetuned = False

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(2)

file_lst = "file/test.txt"
rgb_mean = np.array([123.68, 116.779, 103.939])
num_class = 10
file_class = "file/class.txt"
file_mface = "file/model3d.txt"


def main():
    f_class = open(file_class, 'r')
    count = 1
    class_names = {}
    class_names[0] = "background"
    for line in f_class:
        class_names[line.strip('\n')] = count
        class_names[count] = line.strip('\n')
        count += 1

    f_class.close()

    mean_face = np.zeros([num_class, 3], np.float32)
    f_mface = open(file_mface, 'r')
    for line in f_mface:
        line = line.strip('\n').split(' ')
        xyz = np.array([float(line[1]), float(line[2]), float(line[3])])
        mean_face[class_names[line[0]] - 1] = xyz
    f_mface.close()

    f = open(file_lst, 'r')
    vgg16_rpn = fddb_symbol_vgg16.get_vgg16_rpn()
    _, rpn_args, rpn_auxs = mx.model.load_checkpoint(rpn_prefix, rpn_epoch)
    rpn_args, rpn_auxs = init_from_vgg16(ctx, vgg16_rpn, rpn_args, rpn_auxs)

    vgg16_fc = fddb_symbol_fc.get_vgg16_fc()
    if is_finetuned:
        _, fc_args, fc_auxs = mx.model.load_checkpoint(fc_prefix, fc_epoch)
    else:
        _, fc_args, fc_auxs = mx.model.load_checkpoint(rpn_prefix, rpn_epoch)
    fc_args, fc_auxs = init_from_vgg16(ctx, vgg16_fc, fc_args, fc_auxs)

    rpn_args["mean_face"] = mx.nd.array(mean_face, ctx)

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

        out_height, out_width = vgg16_rpn.infer_shape(data=(1, 3, height, width), mean_face=(10, 3))[1][0][2:4]
        print height, width, out_height, out_width

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
        executor.outputs[0].copyto(softmax_output)
        executor.outputs[1].copyto(regression_output)
        executor.outputs[2].copyto(relu_feature_output)
        executor.outputs[3].copyto(face3dproj_output)
        executor.outputs[4].copyto(box_predict_output)
        softmax_output = np.squeeze(softmax_output.asnumpy())
        regression_output = np.squeeze(regression_output.asnumpy())
        argmax_label = np.uint8(softmax_output.argmax(axis=0))
        out_img = np.uint8(softmax_output.argmax(axis=0))
        box_predict_output = box_predict_output.asnumpy()

        ground_truth, tmp_faceness = calc_ground_truth(out_height, out_width, num_classes, regression_output,
                                                       softmax_output, argmax_label)

        fc_args["ground_truth"] = mx.nd.array(ground_truth, ctx)
        fc_args["box_predict"] = mx.nd.array(box_predict_output, ctx)
        fc_args["relu_feature"] = mx.nd.array(relu_feature_output.asnumpy(), ctx)
        executor = vgg16_fc.bind(ctx, fc_args, args_grad=None, grad_req="null", aux_states=fc_auxs)
        executor.forward(is_train=True)
        ell_output = mx.nd.zeros(executor.outputs[0].shape)
        executor.outputs[0].copyto(ell_output)
        ell_output = ell_output.asnumpy()

        box_predict_at_gt = np.zeros((ground_truth.shape[0], 4))
        for i in xrange(ground_truth.shape[0]):
            box_predict_at_gt[i] = np.array(box_predict_output[0,
                                                               int(ground_truth[i, 0] / 2),
                                                               int(ground_truth[i, 1] / 2)])

        bboxes = calc_ellipse(ell_output, box_predict_at_gt)

        index_lst = np.argsort(-tmp_faceness)
        tmp_ellipses = bboxes[index_lst, :]

        ellipses, faceness, num_ellipses = \
            non_maximum_suppression(tmp_ellipses, tmp_faceness[index_lst], nms_iou, nms_ios)

        f = open("file/bbox.txt", 'w')
        tempt = np.zeros((out_height, out_width, 4), dtype=np.uint8)
        for i in xrange(num_ellipses):
            f.write("%f %f %f %f %f %f\n" %
                    (ellipses[i, 0], ellipses[i, 1], ellipses[i, 2],
                     ellipses[i, 3], ellipses[i, 4], faceness[i]))

        for p in range(0, out_height):
            for q in range(0, out_width):
                pd = out_img[p, q]
                tempt[p, q, 0:3] = np.uint8(pallete[pd * 3: (pd + 1) * 3])
                tempt[p, q, 3] = np.uint8(softmax_output[pd, p, q] * 255)
		#print tempt

        f.close()
        out_img = Image.fromarray(tempt)
        out_img.save("a.png")

    f.close()


if __name__ == "__main__":
    main()