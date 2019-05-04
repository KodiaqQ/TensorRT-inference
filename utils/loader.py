from keras.models import load_model


class Loader(object):
    def __init__(self, type):
        self.type = type
        self.load_by_types = {
            'Keras': self.__load_keras,
            'Tensorflow': self.__load_tf
        }

    def load(self, path):
        try:
            model = self.load_by_types[self.type](path)
        except FileExistsError:
            return None

        return model

    def __load_keras(self, path):
        model = load_model(path)
        # model.summary()

        return model

    def __load_tf(self, path):

        return None

    def __load_pytorch(self, path):

        return None
