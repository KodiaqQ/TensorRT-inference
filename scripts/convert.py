from keras import backend as K
from keras_retinanet import models
import tensorflow as tf
import keras


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
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
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


if __name__ == '__main__':
    keras.backend.tensorflow_backend.set_session(get_session())

    model_path = 'resnet50.h5'

    model = models.load_model(model_path, backbone_name='resnet50')
    model.summary()

    # convert2tf
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, '', 'resnet50.pb', as_text=False)
