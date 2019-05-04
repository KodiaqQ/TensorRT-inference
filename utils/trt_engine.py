from tf_engine import TfEngine
from tensorflow.contrib import tensorrt as tftrt
import copy
import numpy as np


class TftrtEngine(TfEngine):
    def __init__(self, graph, batch_size, precision):
        tftrt_graph = tftrt.create_inference_graph(
            graph.frozen,
            outputs=graph.y_name,
            max_batch_size=batch_size,
            max_workspace_size_bytes=1 << 30,
            precision_mode=precision,
            minimum_segment_size=2)

        opt_graph = copy.deepcopy(graph)
        opt_graph.frozen = tftrt_graph
        super(TftrtEngine, self).__init__(opt_graph)
        self.batch_size = batch_size

    def infer(self, x):
        num_tests = x.shape[0]
        y = np.empty((num_tests, 10), np.float32)
        batch_size = self.batch_size

        for i in range(0, num_tests, batch_size):
            x_part = x[i: i + batch_size]
            y_part = self.sess.run(self.y_tensor,
                                   feed_dict={self.x_tensor: x_part})
            y[i: i + batch_size] = y_part
        return y