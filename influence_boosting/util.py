import json
import numpy as np
from os import system

from catboost import CatBoostClassifier


def expand_shapes_to_array(x, first_dim=True):
    if not isinstance(x, np.ndarray):
        x_array = np.array(x)
    else:
        x_array = x
    is_expanded = False
    if len(x_array.shape) == 1:
        if first_dim:
            x_array = np.array([x])
        else:
            x_array = np.array([[x_c] for x_c in x])
        is_expanded = True
    return x_array, is_expanded


def catboost_fit_predict(train_documents, train_targets, test_documents, prediction_type='RawFormulaVal',
                         **catboost_params):
    if 'gpu_ram_part' in catboost_params:
        gpu_ram_part = catboost_params.pop('gpu_ram_part')
    else:
        gpu_ram_part = None
    cbc = CatBoostClassifier(**catboost_params)
    if gpu_ram_part is not None:
        cbc.set_params(gpu_ram_part=gpu_ram_part)
    cbc.fit(train_documents, train_targets)
    return cbc.predict(test_documents, prediction_type)


def export_catboost_to_json(src_path, dst_path):
    command = 'export_catboost %s > %s' % (src_path, dst_path)
    system(command)


def read_json_params(path):
    with open(path) as f:
        return json.load(f)