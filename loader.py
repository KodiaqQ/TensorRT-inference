from keras.models import load_model


class Loader:
    def __init__(self):
        self.LOAD_BY_TYPES = {
            'Keras': self.__load_keras,
            'Tensorflow': self.__load_tf
        }

    def load(self, type, path):
        try:
            model = self.LOAD_BY_TYPES[type](path)
        except FileExistsError:
            return None

        return model

    def __load_keras(self, path):
        model = load_model(path)
        model.summary()

        return model

    def __load_tf(self, path):

        return None

    def __load_pytorch(self, path):

        return None
