import configparser
import re
import modeler
import argparse
from keras.datasets import mnist
import keras.backend as K


def get_test_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_h = x_test.shape[1]
    img_w = x_test.shape[2]

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_h, img_w)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test, y_test


if __name__ == '__main__':

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--c', help='Path to .ini configuration file', default='config.ini')
    args = aparser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.c)

    regex = re.compile(r'Model_\d+')

    model_names = list(filter(regex.search, config.sections()))

    x_test, y_test = get_test_dataset()
    img_h = x_test.shape[1]
    img_w = x_test.shape[2]

    for model_name in model_names:

        model_path = config[model_name]['path']
        model_type = config[model_name]['type']
        height = config[model_name]['height']
        width = config[model_name]['width']
        batch_size = config[model_name]['batch']

        model = modeler.Modeler(type=model_type, name=model_name)

        if model is None:
            print(model_type + ' type is not loaded or supported for ' + model_name)
            continue

        model.infer(data=x_test, path=model_path)

