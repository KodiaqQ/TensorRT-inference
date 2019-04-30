import tensorrt as tftrt
import configparser
import numpy as np
import re
import loader
import argparse


def verify(result, ans):
    num_tests = ans.shape[0]
    error = 0
    for i in range(0, num_tests):
        a = np.argmax(ans[i])
        r = np.argmax(result[i])
        if (a != r): error += 1

    if (error == 0):
        print('PASSED')
    else:
        print('FAILURE')


if __name__ == '__main__':

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--c', help='Path to .ini configuration file', default='config.ini')
    args = aparser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.c)

    regex = re.compile(r'Model_\d+')

    loader = loader.Loader()

    model_names = list(filter(regex.search, config.sections()))

    for model_name in model_names:
        model_path = config[model_name]['path']
        model_type = config[model_name]['type']
        height = config[model_name]['height']
        width = config[model_name]['width']
        batch = config[model_name]['batch']

        model = loader.load(path=model_path, type=model_type)

        if model is None:
            print(model_type + ' type is not loaded or supported for ' + model_name)
            continue
