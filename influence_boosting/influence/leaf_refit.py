from abc import ABCMeta
from copy import deepcopy

import numpy as np

from ..loss import BinaryCrossEntropyLoss
from ..tree.tree import CatBoostTree, CatBoostEnsemble


class CBLeafRefitObliviousTree(CatBoostTree):
    def _initialize_influence_tree(self, train_documents, train_targets, prev_approxes, loss, train_weights,
                                   l2_regularization_coef, learning_rate, leaf_method):
        self._initialize_parameters(loss, l2_regularization_coef, learning_rate, leaf_method)
        self._custom_initialize(train_documents, train_targets, prev_approxes, train_weights)
        self._documents_original_approxes = prev_approxes + self._document_predictions_in_this_tree

    def _initialize_parameters(self, loss_function, l2_regularization_coef, learning_rate, leaf_method):
        self.loss_function = loss_function
        self.l2_reg_coef = l2_regularization_coef
        self.learning_rate = learning_rate
        self.leaf_method = leaf_method

    def _custom_initialize(self, train_documents, train_targets, previous_approxes, train_weights):
        self._document_idxs_for_leaves = self.assign_documents_to_leaves(train_documents)
        self._gradients = self.loss_function.gradient(train_targets, previous_approxes)
        if self.leaf_method == 'Newton':
            self._hessians = self.loss_function.hessian(train_targets, previous_approxes)
        else:
            self._hessians = np.ones_like(train_targets, dtype=np.float64)
        self._sum_gradients_in_leaves, self._sum_hessians_with_l2_in_leaves = [], []
        self._document_predictions_in_this_tree = np.zeros_like(train_targets, dtype=np.float64)
        number_of_target_dimensions = train_targets.shape[-1] if self.loss_function.is_multidimensional else 1
        for leaf_idx in xrange(len(self.leaf_values)):
            idxs_of_documents_in_leaf = sorted(list(self._document_idxs_for_leaves[leaf_idx]))
            self._sum_gradients_in_leaves.append(
                np.sum(train_weights[idxs_of_documents_in_leaf] * self._gradients[idxs_of_documents_in_leaf], axis=0)
            )
            self._sum_hessians_with_l2_in_leaves.append(
                np.sum(train_weights[idxs_of_documents_in_leaf] * self._hessians[idxs_of_documents_in_leaf], axis=0)
                + self.l2_reg_coef * (np.eye(number_of_target_dimensions) if number_of_target_dimensions > 1 else 1)
            )
            leaf_prediction = self._calculate_leaf_formula(self._sum_gradients_in_leaves[-1],
                                                           self._sum_hessians_with_l2_in_leaves[-1])
            #assert np.allclose(leaf_prediction, self.leaf_values[leaf_idx], atol=1e-5), '%s %s' % (
            #leaf_prediction, self.leaf_values[leaf_idx])
            self.leaf_values[leaf_idx] = leaf_prediction
            self._document_predictions_in_this_tree[
                sorted(list(self._document_idxs_for_leaves[leaf_idx]))] = leaf_prediction

    def _calculate_leaf_formula(self, sum_gradients, sum_hessians):
        if self.loss_function.is_multidimensional:
            if self.leaf_method == 'Newton':
                return -self.learning_rate * np.linalg.solve(
                    sum_hessians, sum_gradients
                )
            else:
                return -self.learning_rate * sum_gradients / sum_hessians[0]
        else:
            return -self.learning_rate * sum_gradients / sum_hessians

    def get_leaf_value_update(self, leaf_idx, removed_point_idx, labels_train, updated_leaf_document_idxs,
                              current_updated_approxes):
        num_target_dimensions = len(labels_train[0]) if self.loss_function.is_multidimensional else 1
        removed_point_was_in_leaf = removed_point_idx in self._document_idxs_for_leaves[leaf_idx]
        idxs_without_removed = filter(lambda idx: idx != removed_point_idx, updated_leaf_document_idxs)

        if len(idxs_without_removed) and self.leaf_method == 'Newton':
            changed_hessians_with_l2 = np.sum(self.loss_function.hessian(
                labels_train[idxs_without_removed],
                current_updated_approxes[idxs_without_removed]
            ) - self._hessians[idxs_without_removed], axis=0)
        else:
            changed_hessians_with_l2 = np.zeros([num_target_dimensions] * 2) if num_target_dimensions > 1 else 0
        if removed_point_was_in_leaf:
            changed_hessians_with_l2 -= self._hessians[removed_point_idx]

        if len(idxs_without_removed):
            changed_gradients = np.sum(self.loss_function.gradient(
                labels_train[idxs_without_removed],
                current_updated_approxes[idxs_without_removed]
            ) - self._gradients[idxs_without_removed], axis=0)
        else:
            changed_gradients = np.zeros_like(labels_train[0]) if num_target_dimensions > 1 else 0
        if removed_point_was_in_leaf:
            changed_gradients -= self._gradients[removed_point_idx]

        new_sum_hessians_with_l2 = self._sum_hessians_with_l2_in_leaves[leaf_idx] + changed_hessians_with_l2
        new_sum_gradients = self._sum_gradients_in_leaves[leaf_idx] + changed_gradients
        new_value = self._calculate_leaf_formula(new_sum_gradients, new_sum_hessians_with_l2)
        old_value = self.leaf_values[leaf_idx]
        return new_value - old_value


class CBOneStepLeafRefitEnsemble(CatBoostEnsemble):
    def __init__(self, tree_json_path, train_documents, train_targets, train_weights=None, leaf_method='Newton',
                 learning_rate=0.03, loss_function=None, l2_regularization_coef=3, update_set='SinglePoint',
                 **update_set_params):
        self._tree_class_ = CBLeafRefitObliviousTree
        CatBoostEnsemble.__init__(self, tree_json_path)
        if loss_function is None:
            loss_function = BinaryCrossEntropyLoss()
        if train_weights is None:
            train_weights = np.ones_like(train_targets, dtype=np.float64)
        self._train_targets = train_targets
        self._initialize_influence_trees(train_documents, train_targets, loss_function, train_weights,
                                         l2_regularization_coef, learning_rate, leaf_method)
        self._get_documents_to_update_idxs = self._documents_to_update_method_from_update_set_param(update_set,
                                                                                                     **update_set_params)

    def fit(self, removed_point_idx, destination_model=None):
        if destination_model is None:
            destination_model = deepcopy(self)
        document_deltas = np.zeros_like(self._train_targets, dtype=float)
        for t, (tree, new_tree) in enumerate(zip(self.trees, destination_model.trees)):
            documents_to_update_idxs = self._get_documents_to_update_idxs(t, removed_point_idx, document_deltas) # set
            current_updated_approxes = self.trees[t - 1]._documents_original_approxes + document_deltas \
                if t > 0 else np.zeros_like(document_deltas)
            for leaf_idx in xrange(len(tree.leaf_values)):
                leaf_documents_idxs = tree._document_idxs_for_leaves[leaf_idx]  # set?
                updated_leaf_documents_idxs = sorted(list(documents_to_update_idxs.intersection(leaf_documents_idxs)))
                if len(documents_to_update_idxs) == 0:
                    continue
                value_delta = tree.get_leaf_value_update(leaf_idx, removed_point_idx, self._train_targets,
                                                         updated_leaf_documents_idxs, current_updated_approxes)
                new_tree.leaf_values[leaf_idx] = tree.leaf_values[leaf_idx] + value_delta
                document_deltas[updated_leaf_documents_idxs] += value_delta
        return destination_model

    def _initialize_influence_trees(self, train_documents, train_targets, loss_function, train_weights,
                                    l2_regularization_coef, learning_rate, leaf_method):
        #assert all(isinstance(tree, CBLeafRefitObliviousTree) for tree in self.trees)
        current_approxes = np.zeros_like(train_targets)
        for t, tree in enumerate(self.trees):
            tree._initialize_influence_tree(train_documents, train_targets, current_approxes, loss_function,
                                            train_weights, l2_regularization_coef, learning_rate, leaf_method)
            current_approxes = tree._documents_original_approxes

    def _documents_to_update_method_from_update_set_param(self, update_set, **update_set_params):
        if update_set == 'SinglePoint':
            return self._only_removed_point_idx_set
        elif update_set == 'AllPoints':
            return self._all_points_idxs_set
        elif update_set == 'TopKLeaves':
            k = update_set_params.pop('k')
            return lambda t, idx, d: self._top_k_leaves_idxs_set(k, t, idx, d)
        else:
            raise ValueError('Unknown update set param %s' % update_set)

    def _only_removed_point_idx_set(self, t, removed_point_idx, deltas):
        return {removed_point_idx}

    def _all_points_idxs_set(self, t, removed_point_idx, deltas):
        return set(range(len(self._train_targets)))

    def _top_k_leaves_idxs_set(self, k, t, removed_point_idxs, deltas):
        tree = self.trees[t]
        leaf_deltas = [
            np.sum(np.abs(deltas[list(tree._document_idxs_for_leaves[leaf_idx])]))
            for leaf_idx in xrange(len(tree.leaf_values))
        ]
        leaf_idxs_to_update = np.argsort(leaf_deltas)[-k:]
        return reduce(lambda x, y: x | y, (tree._document_idxs_for_leaves[l] for l in leaf_idxs_to_update)) | {removed_point_idxs}

    @property
    def _tree_class(self):
        return self._tree_class_