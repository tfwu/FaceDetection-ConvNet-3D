import numpy as np
import mxnet as mx
import logging
from collections import namedtuple
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric

BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])


def print_accuracy(softmax_count, f, class_names=None, bgfg=False):
    if not bgfg:
        for i in range(0, 11):
            if softmax_count[i, 2] == 0:
                continue
            print class_names[i], ':\t', \
                int(softmax_count[i, 2]), '\t', float(softmax_count[i, 0]) / float(softmax_count[i, 2])
            f.write(
                    "%s\t%d\t%f\n" % (
                        class_names[i], int(softmax_count[i, 2]),
                        float(softmax_count[i, 0]) / float(softmax_count[i, 2])))
    else:
        print 'bg', ':\t', int(softmax_count[0, 2]), '\t', float(softmax_count[0, 0]) / float(softmax_count[0, 2])
        print 'fg', ':\t', int(softmax_count[1, 2]), '\t', float(softmax_count[1, 0]) / float(softmax_count[1, 2])


def get_accuracy(softmax_count, bgfg=False):
    accuracy = 0
    count = 0
    if not bgfg:
        for i in range(0, 11):
            accuracy += softmax_count[i, 0]
            count += softmax_count[i, 2]
    else:
        accuracy = softmax_count[0, 0] + softmax_count[1, 0]
        count = softmax_count[0, 2] + softmax_count[1, 2]

    return accuracy / count


def softmax_metric_vis(label, pred):
    pred = pred.asnumpy()
    print [np.argmax(pred[i, :]) for i in xrange(20)]
    print [int(label[i]) for i in xrange(20)]
    count = np.zeros(3)
    # count[0]: number of instance; count[1]: number of negative instance; count[2]: number of correct
    size = pred.shape[0]
    for i in xrange(0, size):
        max_i = np.argmax(pred[i, :])
        count[0] += 1
        if label[i] == 0:
            count[1] += 1
        if max_i == label[i]:
            count[2] += 1
    return count

def bbox_predict_metric(label, pred):
    res = np.array([.0, .0])
    len = label.shape[0]
    print label[0]
    print pred[0]
    for i in xrange(len):
        for j in xrange(4):
            res[0] += np.abs(pred[i, j] - label[i, j])
        res[1] += np.abs(label[i, 4] - pred[i, 4])
    return np.array([res[0] / len / 4, res[1] / len])


class Solver(object):
    def __init__(self, symbol,
                 ctx=None,
                 begin_epoch=0, num_epoch=0,
                 arg_params=None, aux_params=None, bgfg=False,
                 optimizer='sgd', **kwargs):
        self.symbol = symbol
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.optimizer = optimizer
        self.kwargs = kwargs.copy()
        self.bgfg = bgfg

    def fit(self, train_data, eval_data=None,
            eval_metric='acc',
            grad_req='write',
            logger=None,
            softmax_metric=None,
            regression_metric=None,
            epoch_end_callback=None):

        f = open("log_rpn.txt", 'w')
        if logger is None:
            logger = logging
        logging.info('Start training with %s', str(self.ctx))
        f.write('Start training with %s\n' % str(self.ctx))
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=(1, 3, 128, 128), mean_face=(10, 3),
                                                                     ground_truth=(10, 2), bbox_label=(10, 5))
        arg_names = self.symbol.list_arguments()
        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith("mean_face") or name.endswith('cls_label') or
                        name.endswith('proj_weight') or name.endswith('proj_label') or name.endswith('ground_truth') or
                        name.endswith('bbox_label') or name.endswith("bbox_weight")):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        else:
            self.grad_params = None

        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

        data_name = train_data.data_name
        cls_label_name = train_data.cls_label_name
        proj_label_name = train_data.proj_label_name
        proj_weight_name = train_data.proj_weight_name
        ground_truth_name = train_data.ground_truth_name
        bbox_label_name = train_data.bbox_label_name
        bbox_weight_name = train_data.bbox_weight_name

        self.optimizer = opt.create(self.optimizer, rescale_grad=1.0, **(self.kwargs))
        self.updater = get_updater(self.optimizer)
        eval_metric = metric.create(eval_metric)

        for epoch in range(self.begin_epoch, self.num_epoch):
            if eval_data:
                logger.info(" in eval process...")
                f.write(" in eval process...")
                nbatch = 0
                softmax_proj = np.zeros((11, 3))
                proj_regression_loss = .0
                bbox_predict_loss = np.array([.0, .0])
                eval_data.reset()
                for data in eval_data:
                    nbatch += 1
                    print "Eval batch:", nbatch
                    softmax_shape = data[cls_label_name].shape
                    self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
                    self.arg_params[cls_label_name] = mx.nd.array(
                            data[cls_label_name].reshape((softmax_shape[0], softmax_shape[1] * softmax_shape[2])),
                            self.ctx)
                    self.arg_params[proj_label_name] = mx.nd.array(data[proj_label_name], self.ctx)
                    self.arg_params[proj_weight_name] = mx.nd.array(data[proj_weight_name], self.ctx)
                    self.arg_params[ground_truth_name] = mx.nd.array(data[ground_truth_name], self.ctx)
                    self.arg_params[bbox_label_name] = mx.nd.array(data[bbox_label_name], self.ctx)
                    self.arg_params[bbox_weight_name] = mx.nd.array(data[bbox_weight_name], self.ctx)
                    self.arg_params["mean_face"] = mx.nd.array(train_data.mean_face, self.ctx)

                    executor = self.symbol.bind(self.ctx, self.arg_params,
                                                args_grad=self.grad_params,
                                                grad_req=grad_req,
                                                aux_states=self.aux_params)

                    softmax_output_array = mx.nd.zeros(executor.outputs[0].shape)
                    proj_regression_output_array = mx.nd.zeros(executor.outputs[1].shape)
                    bbox_predict_output_array = mx.nd.zeros(executor.outputs[2].shape)
                    ell_label = mx.nd.zeros(executor.outputs[3].shape)
                    bbox_predict = mx.nd.zeros(executor.outputs[4].shape)
                    executor.forward(is_train=True)
                    executor.outputs[0].copyto(softmax_output_array)
                    executor.outputs[1].copyto(proj_regression_output_array)
                    executor.outputs[2].copyto(bbox_predict_output_array)
                    executor.outputs[3].copyto(ell_label)
                    executor.outputs[4].copyto(bbox_predict)

                    softmax_shape = softmax_output_array.shape
                    index_label = np.nonzero(data[cls_label_name]
                                             .reshape(softmax_shape[0], softmax_shape[2] * softmax_shape[3]) - 255)
                    label = mx.nd.array(data[cls_label_name]
                                        .reshape(softmax_shape[0], softmax_shape[2] * softmax_shape[3])
                                        [:, index_label[1]])
                    pred = mx.nd.array((softmax_output_array.asnumpy()
                                        .reshape(softmax_shape[0], softmax_shape[1],
                                                 softmax_shape[2] * softmax_shape[3]))
                                       [..., index_label[1]])
                    if softmax_metric:
                        tempt = softmax_metric(label, pred, 11)
                        softmax_proj += tempt

                    proj_label = data[proj_label_name]
                    proj_weight = data[proj_weight_name]
                    proj_pred = proj_regression_output_array.asnumpy().reshape(data[proj_weight_name].shape)
                    index_nonzero = np.nonzero(data[proj_weight_name])
                    proj_regress_tmp = regression_metric(proj_label[index_nonzero], proj_pred[index_nonzero],
                                                         proj_weight[index_nonzero])
                    proj_regression_loss += proj_regress_tmp

                    bbox_pred = bbox_predict_output_array.asnumpy()
                    bbox_predict_tmp = bbox_predict_metric(ell_label.asnumpy(), bbox_pred)
                    bbox_predict_loss += bbox_predict_tmp

                    print "Validation-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f" % \
                          (epoch, nbatch, get_accuracy(tempt, self.bgfg), proj_regress_tmp,
                           bbox_predict_tmp[0], bbox_predict_tmp[1])
                    f.write("Validation-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f\n" %
                            (epoch, nbatch, get_accuracy(tempt, self.bgfg), proj_regress_tmp,
                             bbox_predict_tmp[0], bbox_predict_tmp[1]))

                    img_info = eval_data.AllImg[nbatch - 1]
                    print "%s\twidth: %d height: %d num_face: %d" % \
                          (img_info.filename, img_info.width, img_info.height, img_info.num_faces)
                    f.write("%s\twidth: %d height: %d num_face: %d\n" %
                            (img_info.filename, img_info.width, img_info.height, img_info.num_faces))

                    executor.outputs[0].wait_to_read()
                    executor.outputs[1].wait_to_read()
                    executor.outputs[2].wait_to_read()
                    executor.outputs[3].wait_to_read()

                print_accuracy(softmax_proj, f, train_data.class_names, self.bgfg)
                logger.info("ALL Validation accuracy: %f", get_accuracy(softmax_proj, self.bgfg))
                logger.info('Validation projection regression: %f', proj_regression_loss / nbatch)
                logger.info('Validation bbox predict: %f %f', bbox_predict_loss[0] / nbatch,
                            bbox_predict_loss[1] / nbatch)
                f.write("ALL Validation accuracy: %f\n" % get_accuracy(softmax_proj, self.bgfg))
                f.write("Validation projection regression: %f\n" % (proj_regression_loss / nbatch))
                f.write("Validation bbox predict: %f %f\n" % (bbox_predict_loss[0] / nbatch,
                                                              bbox_predict_loss[1] / nbatch))

            nbatch = 0
            train_data.reset()
            eval_metric.reset()
            proj_regress_loss_t = .0
            proj_regress_loss_b = .0
            softmax_count = np.zeros((11, 3))
            softmax_batch = np.zeros((11, 3))
            bbox_predict_loss_t = np.array([.0, .0])
            bbox_predict_loss_b = np.array([.0, .0])
            for data in train_data:
                nbatch += 1
                softmax_shape = data[cls_label_name].shape
                self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
                self.arg_params[cls_label_name] = mx.nd.array(
                        data[cls_label_name].reshape((softmax_shape[0], softmax_shape[1] * softmax_shape[2])), self.ctx)
                self.arg_params[proj_label_name] = mx.nd.array(data[proj_label_name], self.ctx)
                self.arg_params[proj_weight_name] = mx.nd.array(data[proj_weight_name], self.ctx)
                self.arg_params[ground_truth_name] = mx.nd.array(data[ground_truth_name], self.ctx)
                self.arg_params[bbox_label_name] = mx.nd.array(data[bbox_label_name], self.ctx)
                self.arg_params[bbox_weight_name] = mx.nd.array(data[bbox_weight_name], self.ctx)
                self.arg_params["mean_face"] = mx.nd.array(train_data.mean_face, self.ctx)

                self.executor = self.symbol.bind(self.ctx, self.arg_params,
                                                 args_grad=self.grad_params,
                                                 grad_req=grad_req,
                                                 aux_states=self.aux_params)
                assert len(self.symbol.list_arguments()) == len(self.executor.grad_arrays)

                update_dict = {name: nd for name, nd in
                               zip(self.symbol.list_arguments(), self.executor.grad_arrays) if nd}
                output_dict = {}
                output_buff = {}
                for key, arr in zip(self.symbol.list_outputs(), self.executor.outputs):
                    output_dict[key] = arr
                    output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
                self.executor.forward(is_train=True)
                for key in output_dict:
                    output_dict[key].copyto(output_buff[key])
                self.executor.backward()

                '''
                for i in xrange(0, 49):
                    if self.executor.grad_arrays[i] != None:
                        print i, arg_names[i], self.executor.grad_arrays[i].asnumpy()[0]
                '''

                for key, arr in update_dict.items():
                    if key != 'upsample_proposal_weight':
                        self.updater(key, arr, self.arg_params[key])
                        '''
                        if key == 'config_fc1_weight':
                            print 'config_fc1_weight'
                            print 'param:', self.arg_params[key].asnumpy()
                            print 'grad:', self.executor.grad_arrays[39].asnumpy()
                        if key == 'refine_proj_param_weight':
                            print 'refine_proj_param_weight'
                            print 'param:', self.arg_params[key].asnumpy()
                            print 'grad:', self.executor.grad_arrays[47].asnumpy()
                        '''

                pred_shape = self.executor.outputs[0].shape
                index_label = np.nonzero(data[cls_label_name]
                                         .reshape(softmax_shape[0], softmax_shape[1] * softmax_shape[2]) - 255)
                label = mx.nd.array(data[cls_label_name].reshape(softmax_shape[0], softmax_shape[1] * softmax_shape[2])
                                    [:, index_label[1]])
                pred = mx.nd.array((output_buff["proposal_cls_loss_output"].asnumpy()
                                    .reshape(pred_shape[0], pred_shape[1], pred_shape[2] * pred_shape[3]))
                                   [..., index_label[1]])
                if softmax_metric:
                    tempt = softmax_metric(label, pred, 11)
                    softmax_count += tempt
                    softmax_batch += tempt

                # for q in range(0, 50):
                #    print label.asnumpy()[0, q], ':', pred.asnumpy()[0, 0, q], pred.asnumpy()[0, 1, q]

                proj_label = data[proj_label_name]
                proj_weight = data[proj_weight_name]
                proj_pred = output_buff["proj_regression_loss_output"].asnumpy()\
                    .reshape(data[proj_weight_name].shape)
                index_nonzero = np.nonzero(data[proj_weight_name])
                proj_regress_tmp = regression_metric(proj_label[index_nonzero], proj_pred[index_nonzero],
                                                     proj_weight[index_nonzero])
                proj_regress_loss_t += proj_regress_tmp
                proj_regress_loss_b += proj_regress_tmp

                ell_label = output_buff["ell_label_output"].asnumpy()
                bbox_pred = output_buff["ellipse_predict_loss_output"].asnumpy()
                bbox_predict_tmp = bbox_predict_metric(ell_label, bbox_pred)
                bbox_predict_loss_t += bbox_predict_tmp
                bbox_predict_loss_b += bbox_predict_tmp

                self.executor.outputs[0].wait_to_read()
                self.executor.outputs[1].wait_to_read()
                self.executor.outputs[2].wait_to_read()
                self.executor.outputs[3].wait_to_read()

                print "Training-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f" % \
                      (epoch, nbatch, get_accuracy(tempt, self.bgfg), proj_regress_tmp,
                       bbox_predict_tmp[0], bbox_predict_tmp[1])
                f.write("Training-epoch[%d]-batch[%d]: acc:%f\tproj_regress:%f\tbbox_regress:%f\tbbox_angle:%f\n" %
                        (epoch, nbatch, get_accuracy(tempt, self.bgfg), proj_regress_tmp,
                         bbox_predict_tmp[0], bbox_predict_tmp[1]))

                img_info = train_data.AllImg[nbatch - 1]
                print "%s\twidth: %d height: %d num_face: %d" % \
                      (img_info.filename, img_info.width, img_info.height, img_info.num_faces)
                f.write("%s\twidth: %d height: %d num_face: %d\n" % \
                        (img_info.filename, img_info.width, img_info.height, img_info.num_faces))

                if nbatch % 50 == 0:
                    print_accuracy(softmax_batch, f, train_data.class_names, self.bgfg)
                    softmax_batch = np.zeros((11, 3))
                    print "Keypoints projection regression smoothl1 loss:\t", proj_regress_loss_b / 50
                    f.write("Keypoints projection regression smoothl1 loss:\t%f\n" % (proj_regress_loss_b / 50))
                    print "Bounding box regression:\t", bbox_predict_loss_b / 50
                    f.write("Bounding box regression: %f %f\n" % (bbox_predict_loss_b[0] / 50,
                                                                  bbox_predict_loss_b[1] / 50))
                    #print "Keypoints offset regression smoothl1 loss:\t", offset_regress_loss_b / 50
                    #f.write("Keypoints offset regression smoothl1 loss:\t%f\n" % (offset_regress_loss_b / 50))
                    #print "Keypoints visibility accuracy:\t", float(softmax_vis_batch[2]) / float(softmax_vis_batch[0])
                    #f.write("Keypoints visibility accuracy:\t%f\n" %
                    #        (float(softmax_vis_batch[2]) / float(softmax_vis_batch[0])))
                    softmax_vis_batch = np.zeros(3)
                    proj_regress_loss_b = .0
                    offset_regress_loss_b = .0
                    bbox_predict_loss_b = np.array([.0, .0])

                if nbatch % 1000 == 0:
                    if epoch_end_callback != None:
                        epoch_end_callback(epoch * 100000 + nbatch, self.symbol, self.arg_params, self.aux_params)

            name, value = eval_metric.get()
            print_accuracy(softmax_count, f, train_data.class_names, self.bgfg)
            logger.info("--->Epoch[%d] Train-cls-%s=%f", epoch, name, value)
            logger.info("--->Epoch[%d] Train-proj-reg-smoothl1=%f", epoch, proj_regress_loss_t / nbatch)
            logger.info("--->Epoch[%d] Train-bbox-reg-smoothl1=%f, %f", epoch, bbox_predict_loss_t[0] / nbatch,
                        bbox_predict_loss_t[1] / nbatch)
            #logger.info("--->Epoch[%d] Train-offset-reg-smoothl1=%f", epoch, offset_regress_loss_t / nbatch)
            #logger.info("--->Epoch[%d] Train-vis-acc=%f", epoch, float(softmax_vis_count[2]) / float(softmax_vis_count[0]))
            f.write("--->Epoch[%d] Train-cls-%s=%f\n" % (epoch, name, value))
            f.write("--->Epoch[%d] Train-proj-reg-smoothl1=%f\n" % (epoch, proj_regress_loss_t / nbatch))
            f.write("--->Epoch[%d] Train-bbox-reg-smoothl1=%f, %f" % (epoch, bbox_predict_loss_t[0] / nbatch,
                    bbox_predict_loss_t[1] / nbatch))
            #f.write("--->Epoch[%d] Train-offset-reg-smoothl1=%f\n" % (epoch, offset_regress_loss_t / nbatch))
            #f.write("--->Epoch[%d] Train-vis-acc=%f" % (epoch, float(softmax_vis_count[2]) / float(softmax_vis_count[0])))

        f.close()
