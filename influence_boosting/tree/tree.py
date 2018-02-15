import json
import numpy as np
from abc import abstractmethod, ABCMeta
from copy import deepcopy


class BaseSplit(object):
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return self._call_(x)

    @abstractmethod
    def _call_(self, x):
        return np.array([])

    @abstractmethod
    def is_close(self, other_split, rtol=1.e-5, atol=1.e-8, equal_nan=False):
        return False


class NumericFeatureSplit(BaseSplit):
    def __init__(self, feature_idx, border):
        self.feature_idx = feature_idx
        self.border = border

    def _call_(self, x):
        return x[:, self.feature_idx] > self.border

    def is_close(self, other_split, rtol=1.e-5, atol=1.e-8, equal_nan=False):
        return self.feature_idx == other_split.feature_idx and np.allclose(self.border, other_split.border,
                                                                           rtol, atol, equal_nan)


class BaseObliviousTree(object):
    __metaclass__ = ABCMeta

    def get_leaf_idx(self, x):
        leaf_idx = np.zeros(len(x))
        for i, split in enumerate(reversed(self.splits)):
            # Shift left + bit corresponding to i_th split
            leaf_idx = 2 * leaf_idx + split(x)
        return leaf_idx.astype(int)

    def __call__(self, x):
        leaf_idxs = self.get_leaf_idx(x)
        if isinstance(self.leaf_values[0], np.ndarray):
            result = self.leaf_values[leaf_idxs, :]
        else:
            result = self.leaf_values[leaf_idxs]
        return result

    def assign_documents_to_leaves(self, train_documents):
        leaf_to_document_idxs = []
        train_documents_leaf_idxs = self.get_leaf_idx(train_documents)
        for leaf_idx in xrange(len(self.leaf_values)):
            train_documents_from_leaf_idxs = list(np.where(train_documents_leaf_idxs == leaf_idx)[0])
            leaf_to_document_idxs.append(set(train_documents_from_leaf_idxs))
        return leaf_to_document_idxs

    def is_close(self, other_tree, check_structure=True, check_leaves=True,
                 rtol=1.e-5, atol=1.e-8, equal_nan=False):
        if check_structure:
            is_structure_close = all(our_split.is_close(other_split, rtol, atol, equal_nan)
                                     for our_split, other_split in zip(self.splits, other_tree.splits))
        else:
            is_structure_close = True
        if check_leaves:
            are_leaves_close = np.allclose(self.leaf_values, other_tree.leaf_values, rtol, atol, equal_nan)
        else:
            are_leaves_close = True
        return is_structure_close and are_leaves_close

    @property
    @abstractmethod
    def splits(self):
        return []

    @property
    @abstractmethod
    def leaf_values(self):
        return np.array([])

    @leaf_values.setter
    @abstractmethod
    def leaf_values(self, v):
        pass


class CatBoostTree(BaseObliviousTree):
    def __init__(self, data_json):
        self._read(data_json)

    def _read(self, data_json):
        CatBoostTree._validate_data(data_json)
        self._splits = [NumericFeatureSplit(split_dict['feature_idx'], split_dict['border'])
                        for split_dict in data_json['splits']]
        self._leaf_values = np.array(data_json['leaf_values'])

    @staticmethod
    def _validate_data(data_json):
        for split in data_json['splits']:
            assert isinstance(split['feature_idx'], int)
            assert isinstance(split['border'], (int, float))
        for value in data_json['leaf_values']:
            assert isinstance(value, (int, float, list, tuple))
        num_splits = len(data_json['splits'])
        num_values = len(data_json['leaf_values'])
        assert num_values == 2 ** num_splits

    @classmethod
    def copy_tree_with_replaced_leaf_values(cls, tree, leaf_values):
        assert len(tree.leaf_values) == len(leaf_values)
        new_tree = cls.__new__(cls)
        new_tree._splits = deepcopy(tree.splits)
        new_tree.leaf_values = leaf_values
        return new_tree

    @property
    def splits(self):
        return self._splits

    @property
    def leaf_values(self):
        return self._leaf_values

    @leaf_values.setter
    def leaf_values(self, v):
        self._leaf_values = v


class BaseTreeEnsemble(object):
    __metaclass__ = ABCMeta

    def __call__(self, x):
        return reduce(lambda a, b: a + b, (tree(x) for tree in self.trees)) + self.bias

    def is_close(self, other_ensemble, check_structure=True, check_leaves=True,
                 rtol=1.e-5, atol=1.e-8, equal_nan=False):
        return all(our_tree.is_close(other_tree, check_structure, check_leaves, rtol, atol, equal_nan)
                   for our_tree, other_tree in zip(self.trees, other_ensemble.trees))

    @property
    @abstractmethod
    def bias(self):
        return 0

    @property
    @abstractmethod
    def trees(self):
        return []

    @trees.setter
    @abstractmethod
    def trees(self, value):
        pass

    @property
    @abstractmethod
    def _tree_class(self):
        return BaseObliviousTree


class CatBoostEnsemble(BaseTreeEnsemble):
    def __init__(self, path):
        self._trees = None
        with open(path) as file_handle:
            self._read(file_handle)

    def _read(self, file_handle):
        data_json = json.load(file_handle)
        self._trees = [self._tree_class(tree_data) for tree_data in data_json]

    @property
    def bias(self):
        return 0

    @property
    def trees(self):
        return self._trees

    @trees.setter
    def trees(self, value):
        self._trees = value

    @property
    def _tree_class(self):
        return CatBoostTree
