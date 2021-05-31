'''
*************************IMPORTANT*************************
NOTE TO STUDENTS: You do NOT need to read or modify this file.
*************************IMPORTANT*************************
'''
# pylint: disable = missing-function-docstring, too-many-arguments, line-too-long, unused-argument, invalid-name

import sys
from importlib import import_module

import numpy as np

import utils
import src

#############################################
############# Naive Bayes Tests #############
#############################################

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='simple',
    weight=1,
    expected={0: 2, 1: 2},
    message='Outputs the label counts Naive Bayes learned from training'
)
def fit_bayes_label_count_simple(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, True)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='simple',
    weight=1,
    expected={(0, 0, 0): 2, (1, 0, 0): 1, (1, 1, 0): 1, (0, 1, 1): 2, (1, 0, 1): 1, (1, 1, 1): 1},
    message='Outputs the feature counts Naive Bayes learned from training'
)
def fit_bayes_feature_count_simple(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, False)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='netflix',
    weight=1,
    expected={0: 2269, 1: 2231},
    message='Outputs the label counts Naive Bayes learned from training'
)
def fit_bayes_label_count_netflix(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, True)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='netflix',
    weight=1,
    expected={
        (0, 1, 1): 1497, (1, 1, 1): 1487, (2, 1, 1): 1325, (3, 1, 1): 1349, (4, 0, 1): 621,
        (5, 1, 1): 1354, (6, 1, 1): 1341, (7, 0, 1): 679, (8, 1, 1): 1922, (9, 0, 1): 762,
        (10, 1, 1): 1516, (11, 1, 1): 1476, (12, 1, 1): 1954, (13, 1, 1): 1498, (14, 1, 1): 1136,
        (15, 1, 1): 1151, (16, 0, 1): 558, (17, 0, 1): 649, (18, 0, 1): 563, (0, 0, 0): 940,
        (1, 0, 0): 718, (2, 0, 0): 736, (3, 0, 0): 779, (4, 0, 0): 981, (5, 1, 0): 1356,
        (6, 1, 0): 1412, (7, 1, 0): 1361, (8, 1, 0): 1957, (9, 0, 0): 688, (10, 1, 0): 1507,
        (11, 0, 0): 809, (12, 1, 0): 1987, (13, 1, 0): 1524, (14, 1, 0): 1286, (15, 1, 0): 1260,
        (16, 1, 0): 1726, (17, 1, 0): 1287, (18, 0, 0): 1560, (2, 0, 1): 906, (4, 1, 1): 1610,
        (5, 0, 1): 877, (6, 0, 1): 890, (9, 1, 1): 1469, (15, 0, 1): 1080, (18, 1, 1): 1668,
        (7, 1, 1): 1552, (16, 1, 1): 1673, (17, 1, 1): 1582, (3, 1, 0): 1490, (7, 0, 0): 908,
        (11, 1, 0): 1460, (18, 1, 0): 709, (1, 1, 0): 1551, (4, 1, 0): 1288, (5, 0, 0): 913,
        (6, 0, 0): 857, (9, 1, 0): 1581, (16, 0, 0): 543, (1, 0, 1): 744, (2, 1, 0): 1533,
        (13, 0, 0): 745, (11, 0, 1): 755, (13, 0, 1): 733, (15, 0, 0): 1009, (10, 0, 0): 762,
        (17, 0, 0): 982, (14, 0, 0): 983, (8, 0, 0): 312, (0, 1, 0): 1329, (0, 0, 1): 734,
        (3, 0, 1): 882, (10, 0, 1): 715, (14, 0, 1): 1095, (8, 0, 1): 309, (12, 0, 1): 277,
        (12, 0, 0): 282
    },
    message='Outputs the feature counts Naive Bayes learned from training'
)
def fit_bayes_feature_count_netflix(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, False)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='ancestry',
    weight=1,
    message='Outputs the label counts Naive Bayes learned from training'
)
def fit_bayes_label_count_ancestry(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, True)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='ancestry',
    weight=1,
    message='Outputs the feature counts Naive Bayes learned from training'
)
def fit_bayes_feature_count_ancestry(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, False)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='heart',
    weight=1,
    message='Outputs the label counts Naive Bayes learned from training'
)
def fit_bayes_label_count_heart(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, True)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='heart',
    weight=1,
    message='Outputs the feature counts Naive Bayes learned from training'
)
def fit_bayes_feature_count_heart(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels, False)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='simple',
    weight=1,
    expected=1.0,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_no_laplace_simple(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': True},
    dataset_name='simple',
    weight=1,
    expected=1.0,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_laplace_simple(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='netflix',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_no_laplace_netflix(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': True},
    dataset_name='netflix',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_laplace_netflix(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='ancestry',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_no_laplace_ancestry(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': True},
    dataset_name='ancestry',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_laplace_ancestry(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='heart',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_no_laplace_heart(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': True},
    dataset_name='heart',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_bayes_laplace_heart(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

#############################################
########### Naive Bayes Questions ###########
#############################################

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='netflix',
    weight=1,
    expected=0.49577777777777776,
)
def eval_question_nb_a(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.nb_questions', '.'), 'question_nb_a')
    return f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='ancestry',
    weight=4,
)
def eval_question_nb_b(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.nb_questions', '.'), 'question_nb_b')
    return f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='ancestry',
    weight=4,
)
def eval_question_nb_c(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.nb_questions', '.'), 'question_nb_c')
    return f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})

@utils.question_part(
    classifier='naive_bayes',
    parameters={'use_laplace_add_one': False},
    dataset_name='netflix',
    weight=4,
)
def eval_question_nb_d(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.nb_questions', '.'), 'question_nb_d')
    return sorted(list(f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})))

#############################################
######### Logistic Regression Tests #########
#############################################

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='simple',
    weight=1,
    expected=np.array([-0.14577434, 0.82004294, -0.06660849]),
    message='Outputs the weights Logistic Regression learned from training'
)
def fit_logistic_simple(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 1500},
    dataset_name='netflix',
    weight=1,
    expected=np.array([
        -1.21519064e+00, 1.43788786e-01, -3.55550372e-02, -1.96166019e-01,
        -1.14173764e-01, 3.25305312e-01, 3.00158721e-02, -1.13328595e-01,
        2.12799561e-01, -4.67992764e-02, -8.72192346e-02, 7.96961369e-02,
        4.61929225e-02, 9.40589329e-03, -1.68137265e-03, -1.08039126e-01,
        5.73541313e-03, -2.19154037e-02, 2.74605127e-01, 1.75370473e+00,
    ]),
    message='Outputs the weights Logistic Regression learned from training'
)
def fit_logistic_netflix(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='ancestry',
    weight=1,
    message='Outputs the weights Logistic Regression learned from training'
)
def fit_logistic_ancestry(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='heart',
    weight=1,
    message='Outputs the weights Logistic Regression learned from training'
)
def fit_logistic_heart(clf, train_features, train_labels, test_features, test_labels, package=None):
    return fitting(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='simple',
    weight=1,
    expected=1.0,
    message='Percentage of correctly labeled answers:'
)
def predict_logistic_simple(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 1500},
    dataset_name='netflix',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_logistic_netflix(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='ancestry',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_logistic_ancestry(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='heart',
    weight=2,
    message='Percentage of correctly labeled answers:'
)
def predict_logistic_heart(clf, train_features, train_labels, test_features, test_labels, package=None):
    return predictions(clf, train_features, train_labels, test_features, test_labels)

#############################################
####### Logistic Regression Questions #######
#############################################

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='heart',
    expected=[8, 11, 13, 16, 17],
    weight=1,
)
def eval_question_lr_a(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.lr_questions', '.'), 'question_lr_a')
    return sorted(list(f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})))

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='heart',
    weight=4,
)
def eval_question_lr_b(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.lr_questions', '.'), 'question_lr_b')
    return f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})

@utils.question_part(
    classifier='logistic_regression',
    parameters={'learning_rate': 0.0001, 'max_steps': 10000},
    dataset_name='ancestry',
    weight=4,
)
def eval_question_lr_c(clf, train_features, train_labels, test_features, test_labels, package=src):
    l = locals()
    f = getattr(import_module(f'{package.__name__}.lr_questions', '.'), 'question_lr_c')
    return f(**{k: l[k] for k in set(f.__code__.co_varnames[:f.__code__.co_argcount])})

#############################################
############ Auxiliary Functions ############
#############################################

#pylint:disable=unused-argument
def fitting(clf, train_features, train_labels, test_features, test_labels, use_nb_labels=None):
    clf.fit(train_features, train_labels)
    return (
        clf.label_counts if use_nb_labels is True
        else clf.feature_counts if use_nb_labels is False
        else clf.weights
    )

def predictions(clf, train_features, train_labels, test_features, test_labels):
    clf.fit(train_features, train_labels)
    result_labels = clf.predict(test_features)
    assert len(test_labels) == len(result_labels)
    return (test_labels == result_labels).mean()
#pylint:enable=unused-argument

if __name__ == '__main__':
    utils.main(sys.argv[1:])
