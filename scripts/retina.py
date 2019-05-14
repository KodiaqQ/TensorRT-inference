import copy
import os
import sys
import time

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras_retinanet import models
from tensorflow.contrib import tensorrt as tftrt
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras.preprocessing import image
sys.path.append(os.path.join(sys.path[0], '../../ssd_keras-master/'))

from models.keras_ssd300 import ssd_300


class TfEngine(object):
    def __init__(self, graph):
        g = tf.Graph()
        with g.as_default():
            x_op, y_op = tf.import_graph_def(
                graph_def=graph.frozen, return_elements=graph.x_name + graph.y_name)
            self.x_tensor = x_op.outputs[0]
            self.y_tensor = y_op.outputs[0]

        config = tf.ConfigProto(gpu_options=
                                tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                                              allow_growth=True))

        self.sess = tf.Session(graph=g, config=config)

    def infer(self, x):
        y = self.sess.run(self.y_tensor,
                          feed_dict={self.x_tensor: x})
        return y


class TftrtEngine(TfEngine):
    def __init__(self, graph, batch_size, precision):
        tftrt_graph = tftrt.create_inference_graph(
            graph.frozen,
            outputs=graph.y_name,
            max_batch_size=1000,
            max_workspace_size_bytes=1 << 30,
            precision_mode=precision,
            minimum_segment_size=2)

        opt_graph = copy.deepcopy(graph)
        opt_graph.frozen = tftrt_graph
        super(TftrtEngine, self).__init__(opt_graph)
        self.batch_size = batch_size

    def infer(self, x):
        y = self.sess.run(self.y_tensor,
                          feed_dict={self.x_tensor: x})
        return y


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class Frozen(object):
    def __init__(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        K.set_learning_phase(0)
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            self.frozen = convert_variables_to_constants(session, input_graph_def,
                                                         output_names, freeze_var_names)
        self.x_name = ['input_1']
        self.y_name = [output_names[0]]


class FrozenGraph(object):
    def __init__(self, model, shape, outs):
        shape = (None, shape[0], shape[1], shape[2])
        x_name = 'input_1'
        with K.get_session() as sess:
            x_tensor = tf.placeholder(tf.float32, shape, x_name)
            K.set_learning_phase(0)
            y_tensor = model(x_tensor)
            y_name = outs[0]
            graph = sess.graph.as_graph_def()
            graph0 = tf.graph_util.convert_variables_to_constants(sess, graph, [y_name])

        self.x_name = [x_name]
        self.y_name = [y_name]
        self.frozen = graph0


def load_pb(path_to_pb):
    class froze:
        frozen = None
        x_name = None
        y_name = None
    result = froze

    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        result.frozen = graph_def
        result.x_name = ['input_1']
        result.y_name = ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3']
        return result


if __name__ == '__main__':
    keras.backend.tensorflow_backend.set_session(get_session())

    img_height = 300
    img_width = 300

    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    weights_path = '../weights/ssd300.h5'

    model.load_weights(weights_path, by_name=True)

    input_images = []

    img = image.load_img('../data/1.jpg', target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    start = time.time()
    y_pred = model.predict(input_images)
    print("\n\nKeras processing time: %s \n\n" % (time.time() - start))

    # convert2tf
    # frozen_graph = Frozen(K.get_session(),
    #                       output_names=[out.op.name for out in model.outputs])

    frozen_graph = FrozenGraph(model, shape=(300, 300, 3), outs=[out.op.name for out in model.outputs])

    # frozen_graph = load_pb('resnet50_new.pb')

    tf_engine = TfEngine(frozen_graph)
    start = time.time()
    tf_engine.infer(input_images)
    print("\n\nTF processing time: %s \n\n" % (time.time() - start))

    tftrt_engine = TftrtEngine(frozen_graph, 1, 'FP32')
    start = time.time()
    tftrt_engine.infer(input_images)
    print("\n\nTRT-FP32 processing time: %s \n\n" % (time.time() - start))

    tftrt_engine = TftrtEngine(frozen_graph, 1, 'FP16')
    start = time.time()
    tftrt_engine.infer(input_images)
    print("\n\nTRT-FP16 processing time: %s \n\n" % (time.time() - start))
