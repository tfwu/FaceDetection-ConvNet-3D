import os
import numpy as np
import mxnet as mx
import fddb_symbol_finetune
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
import time

batchsize = 4000
start_epoch = 1
end_epoch = 201

num_cls = 10
channel_len = 64
feature_len = 3136 + 8 + num_cls * (num_cls + 1) + num_cls * channel_len
label_len = 6
ctx = mx.gpu(2)
file_feature = '/home/yunzhu/face_ext/mxnet/AFLW/fddb_feature/feature-'

rpn_prefix = "model_vgg16/VGG16"
finetune_prefix = "model_finetune/finetune"
load_epoch = 813001
retrain = True

num_feature_fold = {}
feature_fold = {}
label_fold = {}
weight_fold = {}


def calc_mean_var(feature_fold):
    target_index = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    mean = np.zeros(feature_len, dtype=np.float32)
    num_feature = 0
    for index in target_index:
        for i in xrange(feature_fold[index].shape[0]):
            mean += feature_fold[index][i]
        num_feature += feature_fold[index].shape[0]
    mean /= float(num_feature)

    var = np.zeros(feature_len, dtype=np.float32)
    for index in target_index:
        for i in xrange(feature_fold[index].shape[0]):
            tmp = feature_fold[index][i] - mean
            var += tmp ** 2
    var = np.sqrt(var / float(num_feature))

    return mean, var


def bbox_predict_metric(label, pred, weight):
    res = np.array([.0, .0, .0])
    len = label.shape[0]
    cnt_pos = 0
    for i in xrange(len):
        if weight[i, 0] != 0:
            for j in xrange(4):
                res[0] += np.abs(pred[i, j] - label[i, j])
            res[1] += np.abs(label[i, 4] - pred[i, 4])
            cnt_pos += 1
        res[2] += np.abs(label[i, 5] - pred[i, 5])
    return np.array([res[0] / cnt_pos / 4, res[1] / cnt_pos, res[2] / len])


def fddb_finetune_fold(fold_index):
    target_index = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    num_train_feature = 0
    num_valid_feature = 0
    for index in target_index:
        if index != fold_index:
            num_train_feature += num_feature_fold[index]
        else:
            num_valid_feature += num_feature_fold[index]

    train_feature = np.zeros((num_train_feature, feature_len), dtype=np.float)
    train_label = np.zeros((num_train_feature, label_len), dtype=np.float)
    train_weight = np.zeros((num_train_feature, label_len), dtype=np.float)
    train_feature_index = 0
    valid_feature = np.zeros((num_valid_feature, feature_len), dtype=np.float)
    valid_label = np.zeros((num_valid_feature, label_len), dtype=np.float)
    valid_weight = np.zeros((num_valid_feature, label_len), dtype=np.float)
    valid_feature_index = 0
    for index in target_index:
        for i in xrange(num_feature_fold[index]):
            if index != fold_index:
                train_feature[train_feature_index] = feature_fold[index][i]
                train_label[train_feature_index] = label_fold[index][i]
                train_weight[train_feature_index] = weight_fold[index][i]
                train_feature_index += 1
            else:
                valid_feature[valid_feature_index] = feature_fold[index][i]
                valid_label[valid_feature_index] = label_fold[index][i]
                valid_weight[valid_feature_index] = weight_fold[index][i]
                valid_feature_index += 1

    if retrain:
        symbol_finetune = fddb_symbol_finetune.get_vgg16_finetune()
        args = {}
        auxs = {}
        arg_names = symbol_finetune.list_arguments()
        aux_names = symbol_finetune.list_auxiliary_states()
        arg_shapes, _, aux_shapes = symbol_finetune.infer_shape(data=(batchsize, feature_len))
        for name, shape in zip(arg_names, arg_shapes):
            if len(shape) < 1:
                continue
            fan_in, fan_out = np.prod(shape[1:]), shape[0]
            factor = fan_in
            scale = np.sqrt(2.34 / factor)
            tempt = np.random.uniform(-scale, scale, size=shape)
            args[name] = mx.nd.array(tempt, ctx)

        for name, shape in zip(aux_names, aux_shapes):
            if len(shape) < 1:
                continue
            fan_in, fan_out = np.prod(shape[1:]), shape[0]
            factor = fan_in
            scale = np.sqrt(2.34 / factor)
            tempt = np.random.uniform(-scale, scale, size=shape)
            auxs[name] = mx.nd.array(tempt, ctx)
    else:
        symbol_finetune = fddb_symbol_finetune.get_vgg16_finetune()
        _, args, auxs = mx.model.load_checkpoint(rpn_prefix, load_epoch)
        for k, v in args.items():
            if v.context != ctx:
                args[k] = mx.nd.zeros(v.shape, ctx)
                v.copyto(args[k])
        for k, v in auxs.items():
            if v.context != ctx:
                auxs[k] = mx.nd.zeros(v.shape, ctx)
                v.copyto(auxs[k])
        arg_names = symbol_finetune.list_arguments()
        arg_shapes, _, aux_shapes = symbol_finetune.infer_shape(data=(batchsize, feature_len))

    grad_params = {}
    for name, shape in zip(arg_names, arg_shapes):
        if not (name.endswith('ell_label') or name.endswith('bbox_weight') or name.endswith('data')):
            grad_params[name] = mx.nd.zeros(shape, ctx)

    num_train_batch = num_train_feature / batchsize
    lr = 0.03
    lr_decay = 0.33
    epoch_end_callback = mx.callback.do_checkpoint(finetune_prefix + "-" + fold_index)

    for j in range(start_epoch, end_epoch):
        bbox_predict_loss = np.array([.0, .0, .0])
        if j % 50 == 0 or j == start_epoch:
            lr *= lr_decay
            optimizer = opt.create('sgd',
                                   rescale_grad=1.0 / batchsize,
                                   learning_rate=lr,
                                   momentum=0.9,
                                   wd=0.00001)
            updater = get_updater(optimizer)
        for i in range(num_train_batch):
            feature_b = train_feature[i * batchsize: (i + 1) * batchsize, :]
            label_b = train_label[i * batchsize: (i + 1) * batchsize, :]
            weight_b = train_weight[i * batchsize: (i + 1) * batchsize, :]
            args["data"] = mx.nd.array(feature_b, ctx)
            args["ell_label"] = mx.nd.array(label_b, ctx)
            args["bbox_weight"] = mx.nd.array(weight_b, ctx)
            executor = symbol_finetune.bind(ctx, args,
                                            args_grad=grad_params,
                                            grad_req='write',
                                            aux_states=auxs)
            assert len(symbol_finetune.list_arguments()) == len(executor.grad_arrays)

            update_dict = {name: nd for name, nd in
                           zip(symbol_finetune.list_arguments(), executor.grad_arrays) if nd}
            output_dict = {}
            output_buff = {}
            for key, arr in zip(symbol_finetune.list_outputs(), executor.outputs):
                output_dict[key] = arr
                output_buff[key] = mx.nd.zeros(arr.shape, ctx=mx.cpu())
            executor.forward(is_train=True)

            for key in output_dict:
                output_dict[key].copyto(output_buff[key])

            executor.backward()
            for key, arr in update_dict.items():
                updater(key, arr, args[key])

            executor.outputs[0].wait_to_read()

            face_pred = output_buff["ellipse_predict_loss_output"].asnumpy()

            bbox_predict_b = bbox_predict_metric(label_b, face_pred, weight_b)
            bbox_predict_loss += bbox_predict_b

            if i % 10 == 0:
                print "Training-fold[" + \
                      fold_index + \
                      "]-epoch[%d/%d]-batch[%d/%d]: lr:%f\tbbox_regress:%f\tbbox_angle:%f\tiou_regress:%f" % \
                    (j, end_epoch, i, num_train_batch, lr, bbox_predict_b[0], bbox_predict_b[1], bbox_predict_b[2])

        print "ALL Training: bbox_regress:%f\tbbox_angle:%f\tiou_regress:%f" % \
              (bbox_predict_loss[0] / float(num_train_batch), bbox_predict_loss[1] / float(num_train_batch),
               bbox_predict_loss[2] / float(num_train_batch))

        if j % 25 == 0:
            print "Saving the model:", j
            epoch_end_callback(j, symbol_finetune, args, auxs)

        args["data"] = mx.nd.array(valid_feature, ctx)
        args["ell_label"] = mx.nd.array(valid_label, ctx)
        args["bbox_weight"] = mx.nd.array(np.ones((valid_feature.shape[0], label_len), dtype=np.float), ctx)

        executor = symbol_finetune.bind(ctx, args, args_grad=None, grad_req='null', aux_states=auxs)
        output_dict = {}
        output_buff = {}
        for key, arr in zip(symbol_finetune.list_outputs(), executor.outputs):
            output_dict[key] = arr
            output_buff[key] = mx.nd.zeros(arr.shape, ctx=mx.cpu())
        executor.forward(is_train=True)
        for key in output_dict:
            output_dict[key].copyto(output_buff[key])
        executor.outputs[0].wait_to_read()
        face_pred = output_buff["ellipse_predict_loss_output"].asnumpy()

        print valid_label[0]
        print face_pred[0]
        bbox_predict_b = bbox_predict_metric(valid_label, face_pred, valid_weight)

        print "ALL Validation: bbox_regress:%f\tbbox_angle:%f\tiou_regress:%f" % \
              (bbox_predict_b[0], bbox_predict_b[1], bbox_predict_b[2])


def collecting_data():
    target_index = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    for index in target_index:
        num_feature_fold[index] = len(open(file_feature + index + ".txt", 'r').readlines()) / 3

    for index in target_index:
        print "Collecting data:", index
        feature_fold_raw = open(file_feature + index + ".txt", 'r').readlines()
        feature_fold[index] = np.zeros((num_feature_fold[index], feature_len), dtype=np.float)
        label_fold[index] = np.zeros((num_feature_fold[index], label_len), dtype=np.float)
        weight_fold[index] = np.zeros((num_feature_fold[index], label_len), dtype=np.float)

        index_cnt = 0
        for i in xrange(num_feature_fold[index]):
            tmp = feature_fold_raw[i * 3].strip('\n').strip(' ').split(' ')
            feature_fold[index][index_cnt] = np.array(tmp, dtype=np.float)
            tmp = feature_fold_raw[i * 3 + 1].strip('\n').strip(' ').split(' ')
            label_fold[index][index_cnt, :label_len - 1] = np.array(tmp, dtype=np.float)
            tmp = feature_fold_raw[i * 3 + 2].strip('\n').strip(' ').split(' ')
            label_fold[index][index_cnt, label_len - 1] = float(tmp[0])
            if float(tmp[0]) > 0.1:
                weight_fold[index][index_cnt] = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 10.0])
            else:
                weight_fold[index][index_cnt] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
            index_cnt += 1

    if os.path.exists('./mean.npy'):
        mean = np.load('./mean.npy')
        var = np.load('./var.npy')
    else:
        mean, var = calc_mean_var(feature_fold)
        for i in range(feature_len):
            if var[i] == 0:
                var[i] = 1
        np.save('./mean.npy', mean)
        np.save('./var.npy', var)
        print mean.shape, var.shape
        print mean
        print var

    for index in target_index:
        feature_fold[index] = (feature_fold[index] - mean) / var


target_index = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

if __name__ == "__main__":
    collecting_data()
    for i in target_index:
        fddb_finetune_fold(i)
