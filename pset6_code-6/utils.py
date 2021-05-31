'''
*************************IMPORTANT*************************
NOTE TO STUDENTS: You do NOT need to read or modify this file.
*************************IMPORTANT*************************
'''

import os
import sys
from importlib import import_module

import numpy as np

import src

#Obtains dataset directory; assume it's located in the same directory as this script
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
VALID_DATASET_NAMES = sorted(list(set(
    name.split('-', maxsplit=1)[0]
    for name in os.listdir(DATASET_DIR) if name.endswith('txt') and '-' in name
)))

def get_data(input_file):
    '''Obtain the feature and label data from the given file'''
    with open(input_file) as inp:
        num_cols, num_rows = int(inp.readline().strip()), int(inp.readline().strip())
        features = np.empty((num_rows, num_cols), np.uint8)
        labels = np.empty((num_rows,), np.uint8)
        for row, line in enumerate(inp):
            feat, label = line.split(':')
            for col, val in enumerate(feat.split()):
                features[row][col] = np.uint8(val)
            labels[row] = np.uint8(label.strip())
        features.setflags(write=False)
        labels.setflags(write=False)
        return features, labels

DATASETS = {
    dataset_name: (
        *get_data(os.path.join(DATASET_DIR, f'{dataset_name}-train.txt')),
        *get_data(os.path.join(DATASET_DIR, f'{dataset_name}-test.txt'))
    )
    for dataset_name in VALID_DATASET_NAMES
}

def color(msg, colorname):
    '''Color a message'''
    colors = {'red': '\033[91m', 'green': '\033[92m', 'blue': '\033[94m'}
    return f'{colors[colorname]}{msg}\033[0m' if colorname in colors and os.name != 'nt' else msg

def are_equal(res, exp):
    '''Check that inputs are approximately equal'''
    return np.allclose(res, exp) if isinstance(exp, (float, list, np.ndarray)) else res == exp

QUESTION_PARTS = {}
NUM_PUBLIC_TESTS = 0
NUM_PASSED_TESTS = 0
def question_part(*, classifier, parameters, dataset_name, weight,
                  visibility='visible', expected=None, message=''):
    '''Decorator for functions that use classifiers and parsed data'''
    if dataset_name not in VALID_DATASET_NAMES:
        raise Exception(
            f'\'{dataset_name}\' is not valid\nValid datasets include {VALID_DATASET_NAMES}'
        )
    def decor(func):
        def wrapper(display_question=True, package=src):
            if display_question:
                print(f'\n\t{func.__name__}. {message}')

            # Assumes that the module is snake-case (e.g. naive_bayes)
            # and the class it contains is Pascal-case (e.g. NaiveBayes)
            clf_name = ''.join(word.capitalize() for word in classifier.split('_'))
            clf = getattr(import_module(f'{package.__name__}.{classifier}', '.'), clf_name)
            result = func(clf(**parameters), *DATASETS[dataset_name], package=package)

            if display_question:
                if expected is None:
                    print(color(f'HIDDEN (no runtime exceptions)\nresult:   {result}', 'blue'))
                else:
                    global NUM_PUBLIC_TESTS #pylint:disable=global-statement
                    NUM_PUBLIC_TESTS += 1
                    if are_equal(result, expected):
                        print(color('PASSED', 'green'))
                        global NUM_PASSED_TESTS #pylint:disable=global-statement
                        NUM_PASSED_TESTS += 1
                    else:
                        print(color(f'FAILED\nexpected: {expected}\nresult:   {result}', 'red'))
            return result
        wrapper.__weight__ = weight
        wrapper.__visibility__ = visibility
        QUESTION_PARTS[func.__name__] = wrapper
        return wrapper
    return decor

def main(args):
    '''Display question and output'''

    # Remove duplicates
    # `dict` removes duplicates and also preserves insertion order as of python 3.7
    # (`set` randomizes ordering which is undesirable in this case so we use keyset from `dict`)
    argset = {k: None for k in args}.keys()

    invalid_args = argset - QUESTION_PARTS.keys()
    if invalid_args:
        print(f'The following arguments are not valid: {invalid_args}', file=sys.stderr)
        return

    for question_name in QUESTION_PARTS if not argset else argset:
        QUESTION_PARTS[question_name]()
    print(f'''\nPassed {
        color(NUM_PASSED_TESTS, 'green' if NUM_PASSED_TESTS == NUM_PUBLIC_TESTS else 'red')
    } out of {color(NUM_PUBLIC_TESTS, 'green')} publicly available tests''')
