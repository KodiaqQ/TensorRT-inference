import keras.backend as K
import os
import time
from keras.models import load_model
from utils.freezer import Freezer
from utils.tf_engine import TFEngine
from utils.trt_engine import TRTEngine


class Modeler(object):
    def __init__(self, type, name):
        self.type = type
        self.name = name

        # self.engine = {
        #     'Tensorflow': TFEngine
        # }

        self.infer_by_types = {
            'Keras': self.__infer_keras,
            'Tensorflow': self.__infer_tf,
            'PyTorch': self.__infer_pytorch
        }

    def __infer_keras(self, path, data):
        model = load_model(path)
        t0 = time.time()
        y_test = model.predict(data)
        t1 = time.time()
        print('Keras time %s at %s' % ((t1 - t0), self.name))
        y_test = self.__infer_tf(path=path, data=data)
        y_test = self.__infer_tensorrt(path=path, data=data, presicion='FP32')
        return y_test

    def __infer_tf(self, path, data):
        model = load_model(path)
        graph = Freezer(model=model, shape=(data.shape[1], data.shape[2], 1))
        engine = TFEngine(graph=graph)
        t0 = time.time()
        y_test = engine.infer(data)
        t1 = time.time()
        print('Tensorflow time %s at %s' % ((t1 - t0), self.name))
        K.clear_session()
        return y_test

    def __infer_pytorch(self, path):
        return None

    def __infer_tensorrt(self, path, data, presicion):
        model = load_model(path)
        graph = Freezer(model=model, shape=(data.shape[1], data.shape[2], 1))
        engine = TRTEngine(graph=graph, batch_size=1000, precision=presicion)
        t0 = time.time()
        y_test = engine.infer(data)
        t1 = time.time()
        print('TensorRT time %s at %s' % ((t1 - t0), self.name))
        K.clear_session()
        return y_test

    def infer(self, data, path):
        result = self.infer_by_types[self.type](path=path, data=data)
        return result
