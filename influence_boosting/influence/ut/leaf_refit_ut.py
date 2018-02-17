from copy import deepcopy
from logging import getLogger
from os import mkdir
from os.path import isdir
import numpy as np
from catboost import CatBoostClassifier

from ...influence.leaf_refit import CBOneStepLeafRefitEnsemble
from ...loss import BinaryCrossEntropyLoss
from ...util import export_catboost_to_json, read_json_params
from data.adult import read_train_documents_and_one_hot_targets

logger = getLogger('__main__')


def _test_all_points_retraining(method):
    base_dir = 'data/adult/'
    train_documents, train_targets = read_train_documents_and_one_hot_targets(
        base_dir + 'train_data_catboost_format.tsv'
    )
    train_targets = np.argmax(train_targets, axis=1)

    train_dir = base_dir + 'ut_tmp/'
    if not isdir(train_dir):
        mkdir(train_dir)
    cbc_params = read_json_params(base_dir + 'catboost_params.json')
    cbc_params['leaf_estimation_method'] = method
    cbc_params['random_seed'] = 10
    cbc_params['train_dir'] = train_dir
    cbc = CatBoostClassifier(**cbc_params)
    cbc.fit(train_documents, train_targets)
    cbc.save_model(train_dir + 'model.bin', format='cbm')
    export_catboost_to_json(train_dir + 'model.bin', train_dir + 'model.json')
    full_model = CBOneStepLeafRefitEnsemble(train_dir + 'model.json', train_documents, train_targets,
                                            learning_rate=cbc_params['learning_rate'],
                                            loss_function=BinaryCrossEntropyLoss(),
                                            leaf_method=method,
                                            update_set='AllPoints')

    retrained_model_our = deepcopy(full_model)
    count = 0
    for remove_idx in np.random.randint(len(train_documents), size=20):
        print remove_idx
        cbc = CatBoostClassifier(**cbc_params)
        weights = np.ones_like(train_targets, dtype=np.float64)
        weights[remove_idx] = 1e-20
        cbc.fit(train_documents, train_targets, sample_weight=weights)
        cbc.save_model(train_dir + 'model_tmp.bin', format='cbm')
        export_catboost_to_json(train_dir + 'model_tmp.bin', train_dir + 'model_tmp.json')

        train_documents_wo_removed = np.delete(train_documents, remove_idx, axis=0)
        train_targets_wo_removed = np.delete(train_targets, remove_idx, axis=0)
        retrained_model_true = CBOneStepLeafRefitEnsemble(train_dir + 'model_tmp.json', train_documents_wo_removed,
                                                          train_targets_wo_removed,
                                                          learning_rate=cbc_params['learning_rate'],
                                                          loss_function=BinaryCrossEntropyLoss(),
                                                          leaf_method=method,
                                                          update_set='AllPoints')
        full_model.fit(remove_idx, retrained_model_our)
        if not retrained_model_true.is_close(retrained_model_our, check_leaves=False):
            print('Model structure changed for idx %s' % remove_idx)
            count += 1
            continue
        assert retrained_model_true.is_close(retrained_model_our, check_structure=False), remove_idx
        del cbc


def _test_prediction_consistency(method):
    base_dir = 'data/adult/'
    train_documents, train_targets = read_train_documents_and_one_hot_targets(
        base_dir + 'train_data_catboost_format.tsv'
    )
    train_targets = np.argmax(train_targets, axis=1)

    test_documents, test_targets = read_train_documents_and_one_hot_targets(
        base_dir + 'train_data_catboost_format.tsv'
    )

    train_dir = base_dir + 'ut_tmp/'
    if not isdir(train_dir):
        mkdir(train_dir)
    cbc_params = read_json_params(base_dir + 'catboost_params.json')
    cbc_params['leaf_estimation_method'] = method
    cbc_params['random_seed'] = 10
    cbc_params['train_dir'] = train_dir
    cbc = CatBoostClassifier(**cbc_params)
    cbc.fit(train_documents, train_targets)
    cbc.save_model(train_dir + 'model.bin', format='cbm')
    export_catboost_to_json(train_dir + 'model.bin', train_dir + 'model.json')
    full_model = CBOneStepLeafRefitEnsemble(train_dir + 'model.json', train_documents, train_targets,
                                            learning_rate=0.2, loss_function=BinaryCrossEntropyLoss(),
                                            leaf_method=method,
                                            update_set='AllPoints')
    assert np.allclose(full_model(train_documents), cbc.predict(train_documents, prediction_type='RawFormulaVal'),
                       atol=1e-5),\
                       (full_model(train_documents), cbc.predict(train_documents, prediction_type='RawFormulaVal'))
    assert np.allclose(full_model(test_documents), cbc.predict(test_documents, prediction_type='RawFormulaVal'),
                       atol=1e-5)


def test_all_points_retraining_gradient():
    return _test_all_points_retraining('Gradient')


def test_all_points_retraining_newton():
    return _test_all_points_retraining('Newton')


def test_prediction_consistency_gradient():
    return _test_prediction_consistency('Gradient')


def test_prediction_consistency_newton():
    return _test_prediction_consistency('Newton')


if __name__ == '__main__':
    test_prediction_consistency_gradient()
    test_prediction_consistency_newton()
    test_all_points_retraining_gradient()
    test_all_points_retraining_newton()