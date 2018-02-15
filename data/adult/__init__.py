import numpy as np
import pandas as pd


def one_hot_encode_array(x):
    assert isinstance(x, np.ndarray) and len(x.shape) == 1
    x_int = x.astype(int)
    min_label, max_label = np.min(x_int), np.max(x_int)
    result = np.zeros((len(x_int), max_label - min_label + 1))
    result[xrange(len(result)), x_int - min_label] = 1
    return result


def read_train_documents_and_one_hot_targets(path):
    target_and_document_features = pd.read_csv(path, sep='\t', header=None).values
    return target_and_document_features[:, 1:], one_hot_encode_array(target_and_document_features[:, 0])
