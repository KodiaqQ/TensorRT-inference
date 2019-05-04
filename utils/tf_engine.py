import tensorflow as tf


class TFEngine(object):
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
