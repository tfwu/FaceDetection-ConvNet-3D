import os
import argparse
import mxnet as mx
import numpy as np
import logging
import symbol_vgg16
from data import FileIter
from solver import Solver
from init_vgg16 import init_from_vgg16

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(2)
sigma = 1.0
bgfg = False


def smoothl1_metric(label, pred, weight):
    rec = 0.0
    size = label.shape[0]
    sigma2 = sigma * sigma
    for i in xrange(0, size):
        rec += np.abs(pred[i] - label[i]) * weight[i]
        '''
        if np.abs(tmp) < sigma2:
            rec += 0.5 * tmp * tmp * sigma2
        else:
            rec += np.abs(tmp) - 0.5 / sigma2
        '''
    if size == 0:
        return 0
    return rec / size


def softmax_metric(label, pred, num_classes):
    pred = np.swapaxes(pred.asnumpy(), 1, 2)
    label = label.asnumpy()
    count = np.zeros((num_classes, 3))
    size = label.shape[1]
    for i in xrange(0, size):
        max_i = np.argmax(pred[0, i, :])
        label_i = int(label[0, i])
        count[label_i, 2] += 1
        if max_i == label_i:
            count[label_i, 0] += 1
            count[label_i, 1] += 1
    return count


def softmax_metric_vis(label, pred, num_classes):
    pred = pred.asnumpy()
    label = label.asnumpy()
    count = np.zeros(3)
    size = pred.shape[0]
    for i in xrange(0, size):
        max_i = np.argmax(pred[i, :])
        count[0] += 1
        if label[i] == 0:
            count[1] += 1
        if max_i == label[i]:
            count[2] += 1
    return count


def main():

    pretrain_prefix = "model_vgg16/VGG16"
    pretrain_epoch = 813001
    vgg16_rpn = symbol_vgg16.get_vgg16_rpn()
    _, rpn_args, rpn_auxs = mx.model.load_checkpoint(pretrain_prefix, pretrain_epoch)
    rpn_args, rpn_auxs = init_from_vgg16(ctx, vgg16_rpn, rpn_args, rpn_auxs)
    '''
    rpn_prefix = "model_vgg16/VGG16"
    epoch = 813001
    vgg16_rpn, rpn_args, rpn_auxs = mx.model.load_checkpoint(rpn_prefix, epoch)
    '''
    train_dataiter = FileIter(
            rgb_mean=(123.68, 116.779, 103.939),
            file_lst="./file/train.txt",
            class_names="./file/class.txt",
            file_mface="./file/model3d.txt",
            bgfg=bgfg,
            num_data_used=-1
    )
    eval_dataiter = FileIter(
            rgb_mean=(123.68, 116.779, 103.939),
            file_lst="./file/val.txt",
            class_names="./file/class.txt",
            file_mface="./file/model3d.txt",
            bgfg=bgfg,
            num_data_used=1000
    )
    rpn_model = Solver(
            ctx=ctx,
            symbol=vgg16_rpn,
            arg_params=rpn_args,
            aux_params=rpn_auxs,
            begin_epoch=8,
            num_epoch=50,
            learning_rate=0.000001,
            momentum=0.9,
            wd=0.00001,
            bgfg=bgfg
    )
    rpn_model.fit(
            train_data=train_dataiter,
            eval_data=eval_dataiter,
            regression_metric=smoothl1_metric,
            softmax_metric=softmax_metric,
            epoch_end_callback=mx.callback.do_checkpoint("model_vgg16/VGG16")
    )


if __name__ == "__main__":
    main()
