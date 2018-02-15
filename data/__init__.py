import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from copy import deepcopy


class CTRImputer(object):
    def __init__(self, cat_features_idxs=None):
        if cat_features_idxs is None:
            cat_features_idxs = []
        self._cat_features_idxs = cat_features_idxs
        self._cat_feature_ctrs = None

    def fit(self, documents, labels):
        self._cat_feature_ctrs = {}
        for cat_feature_idx in self._cat_features_idxs:
            feature_column_and_labels = pd.DataFrame(pd.concat([documents.iloc[:, cat_feature_idx], labels],
                                                               axis=1)).dropna()
            feature_column_and_labels.columns = range(len(feature_column_and_labels.columns))
            grouped_by_feature = feature_column_and_labels.groupby(feature_column_and_labels.columns[0])
            feature_values_counters_df = grouped_by_feature.agg(len).sort_index()
            feature_values_target_sums_df = grouped_by_feature.agg(lambda x: sum(map(float, x))).sort_index()
            feature_values_ctrs = feature_values_target_sums_df / feature_values_counters_df
            self._cat_feature_ctrs[cat_feature_idx] = {value: ctr_srs.iloc[0]
                                                       for value, ctr_srs in feature_values_ctrs.iterrows()}

    def transform(self, documents):
        result = documents.copy()
        for cat_feature_idx in self._cat_features_idxs:
            feature_column = documents.iloc[:, cat_feature_idx]
            feature_ctrs = self._cat_feature_ctrs[cat_feature_idx]
            values_without_statistics_mask = feature_column.apply(lambda x: x not in feature_ctrs and x is not np.nan)
            unique_values_without_statistics = feature_column.loc[values_without_statistics_mask].unique()

            feature_ctrs_with_default_values = deepcopy(feature_ctrs)
            feature_ctrs_with_default_values.update({v: -1 for v in unique_values_without_statistics})
            result.iloc[:, cat_feature_idx] = result.iloc[:, cat_feature_idx].map(feature_ctrs_with_default_values)
        return result.astype(float)


class CVCTRImputer(object):
    def __init__(self, cat_features_idxs=None, n_folds=10):
        self._cat_features_idxs = cat_features_idxs
        self._n_folds = n_folds
        self._fold_imputers = []
        self._full_imputer = None
        self._fold_document_idxs = []

    def fit(self, train_data, train_labels, stratify=False, seed=None, shuffle=False, group_col=None):
        if group_col is not None:
            fold_iterator = GroupKFold(self._n_folds)
            fold_generator = fold_iterator.split(train_data.values, train_labels, train_data[group_col])
        else:
            if not stratify:
                fold_iterator = KFold(self._n_folds, random_state=seed, shuffle=shuffle)
            else:
                fold_iterator = StratifiedKFold(self._n_folds, random_state=seed, shuffle=shuffle)
            fold_generator = fold_iterator.split(train_data.values, train_labels)
        for fold, (ctr_estimation_idxs, imputation_idxs) in enumerate(fold_generator):
            self._fold_document_idxs.append(imputation_idxs)
            ctri = CTRImputer(self._cat_features_idxs)
            ctri.fit(train_data.iloc[ctr_estimation_idxs, :], train_labels.iloc[ctr_estimation_idxs])
            self._fold_imputers.append(ctri)
        self._full_imputer = CTRImputer(self._cat_features_idxs)
        self._full_imputer.fit(train_data, train_labels)

    def transform_train(self, train_data):
        result = train_data.copy()
        for imputation_idxs, imputer in zip(self._fold_document_idxs, self._fold_imputers):
            result.iloc[imputation_idxs] = imputer.transform(train_data.iloc[imputation_idxs])
        return result

    def transform_test(self, test_data):
        return self._full_imputer.transform(test_data)


class NanImputer(object):
    def __init__(self, cat_features_idxs, default_cat_value='None'):
        self._cat_features_idxs = cat_features_idxs
        self._num_features_idxs = None
        self._num_features_means = None
        self._default_cat_value = default_cat_value

    def fit(self, data, labels):
        self._num_features_idxs = [i for i in range(data.shape[1]) if i not in self._cat_features_idxs]
        self._num_features_means = data.iloc[:, self._num_features_idxs].mean()

    def transform(self, data):
        result = deepcopy(data)
        result.iloc[:, self._num_features_idxs] = result.iloc[:, self._num_features_idxs].fillna(self._num_features_means)
        result.iloc[:, self._cat_features_idxs] = result.iloc[:, self._cat_features_idxs].fillna(self._default_cat_value)
        return result
