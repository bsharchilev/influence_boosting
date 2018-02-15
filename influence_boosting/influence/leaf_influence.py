from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from influence_boosting.influence_boosting.loss import BinaryCrossEntropyLoss
from influence_boosting.influence_boosting.tree.tree import CatBoostTree, CatBoostEnsemble


class CBLeafInfluenceObliviousTree(CatBoostTree):
    __metaclass__ = ABCMeta

    def _initialize_influence_tree(self, train_documents, train_targets, prev_approxes, loss, weights,
                                   l2_regularization_coef, learning_rate):
        self._initialize_parameters(loss, l2_regularization_coef, learning_rate)
        self._custom_initialize(train_documents, train_targets, prev_approxes, weights)
        self._precompute_influence_statistics(weights)

    def _initialize_parameters(self, loss_function, l2_regularization_coef, learning_rate):
        self.loss_function = loss_function
        self.l2_reg_coef = l2_regularization_coef
        self.learning_rate = learning_rate

    def _custom_initialize(self, train_documents, train_targets, prev_approxes, weights):
        self._document_idxs_for_leaves = self.assign_documents_to_leaves(train_documents)
        self._gradients = self.loss_function.gradient(train_targets, prev_approxes)
        self._hessians = self.loss_function.hessian(train_targets, prev_approxes)
        self._thirds = self.loss_function.third(train_targets, prev_approxes)
        self._document_predictions_in_this_tree = np.zeros_like(train_targets, dtype=np.float64)
        self._denominators = np.zeros_like(self.leaf_values, dtype=np.float64)
        for leaf_idx in xrange(len(self.leaf_values)):
            leaf_enumerator = self._leaf_enumerator(leaf_idx, weights)
            leaf_denominator = self._leaf_denominator(leaf_idx, weights)
            leaf_prediction = -leaf_enumerator / leaf_denominator * self.learning_rate
            assert np.allclose(leaf_prediction, self.leaf_values[leaf_idx], atol=1e-5), '%s %s' % (
                leaf_prediction, self.leaf_values[leaf_idx]
            )
            sorted_document_idxs = sorted(list(self._document_idxs_for_leaves[leaf_idx]))
            self.leaf_values[leaf_idx] = leaf_prediction
            self._document_predictions_in_this_tree[sorted_document_idxs] = leaf_prediction
            self._denominators[leaf_idx] = leaf_denominator

    def _precompute_influence_statistics(self, weights):
        self._naive_gradient_addendum = self._get_naive_gradient_addendum()
        self._da_vector_multiplier = self._get_da_vector_multiplier(weights)

    def leaf_derivative(self, leaf_idx, removed_point_idx, updated_leaf_documents_idxs, da):
        grad_enumerator = np.dot(
            da[updated_leaf_documents_idxs], self._da_vector_multiplier[updated_leaf_documents_idxs]
        )
        if removed_point_idx in updated_leaf_documents_idxs:
           grad_enumerator += self._naive_gradient_addendum[removed_point_idx]
        return -grad_enumerator / self._denominators[leaf_idx] * self.learning_rate

    @abstractmethod
    def _leaf_enumerator(self, leaf_idx, weights):
        return np.array([])

    @abstractmethod
    def _leaf_denominator(self, leaf_idx, weights):
        return np.array([])

    @abstractmethod
    def _get_naive_gradient_addendum(self):
        pass

    @abstractmethod
    def _get_da_vector_multiplier(self, weights):
        pass


class CBGradientLeafInfluenceObliviousTree(CBLeafInfluenceObliviousTree):
    def _leaf_enumerator(self, leaf_idx, weights):
        sorted_document_idxs = sorted(list(self._document_idxs_for_leaves[leaf_idx]))
        return sum(weights[sorted_document_idxs] * self._gradients[sorted_document_idxs])

    def _leaf_denominator(self, leaf_idx, weights):
        sorted_document_idxs = sorted(list(self._document_idxs_for_leaves[leaf_idx]))
        return sum(weights[sorted_document_idxs]) + self.l2_reg_coef

    def _get_naive_gradient_addendum(self):
        return self._document_predictions_in_this_tree / self.learning_rate + self._gradients

    def _get_da_vector_multiplier(self, weights):
        return weights * self._hessians


class CBNewtonLeafInfluenceObliviousTree(CBLeafInfluenceObliviousTree):
    def _leaf_enumerator(self, leaf_idx, weights):
        sorted_document_idxs = sorted(list(self._document_idxs_for_leaves[leaf_idx]))
        return sum(weights[sorted_document_idxs] * self._gradients[sorted_document_idxs])

    def _leaf_denominator(self, leaf_idx, weights):
        sorted_document_idxs = sorted(list(self._document_idxs_for_leaves[leaf_idx]))
        return sum(weights[sorted_document_idxs] * self._hessians[sorted_document_idxs]) + self.l2_reg_coef

    def _get_naive_gradient_addendum(self):
        return self._hessians * self._document_predictions_in_this_tree / self.learning_rate + self._gradients

    def _get_da_vector_multiplier(self, weights):
        return weights * (self._document_predictions_in_this_tree / self.learning_rate * self._thirds + self._hessians)


class CBLeafInfluenceEnsemble(CatBoostEnsemble):
    def __init__(self, tree_json_path, train_documents, train_targets, train_weights=None, leaf_method='Newton',
                 learning_rate=0.03, loss_function=None, l2_regularization_coef=3, update_set='SinglePoint',
                 **update_set_params):
        self._tree_class_ = self._tree_class_from_leaf_method_param(leaf_method)
        CatBoostEnsemble.__init__(self, tree_json_path)
        if loss_function is None:
            loss_function = BinaryCrossEntropyLoss()
        if train_weights is None:
            train_weights = np.ones_like(train_targets, dtype=np.float64)
        self._initialize_influence_trees(train_documents, train_targets, loss_function, train_weights,
                                         l2_regularization_coef, learning_rate)
        self._get_documents_to_update_idxs = self._documents_to_update_method_from_update_set_param(update_set,
                                                                                                     **update_set_params)
        self._train_documents_num = len(train_targets)
        self.influence_trees = None

    def fit(self, removed_point_idx, destination_model=None):
        if destination_model is None:
            destination_model = deepcopy(self)
        destination_model.influence_trees = []
        da = np.zeros(self._train_documents_num, dtype=float)
        for t, (tree, new_tree) in enumerate(zip(self.trees, destination_model.trees)):
            documents_to_update_idxs = self._get_documents_to_update_idxs(t, removed_point_idx, da)
            leaf_derivatives = np.zeros_like(new_tree.leaf_values)
            for leaf_idx in xrange(len(tree.leaf_values)):
                leaf_documents_idxs = tree._document_idxs_for_leaves[leaf_idx]
                updated_leaf_documents_idxs = sorted(list(documents_to_update_idxs.intersection(leaf_documents_idxs)))
                leaf_derivative = tree.leaf_derivative(leaf_idx, removed_point_idx, updated_leaf_documents_idxs, da)
                leaf_derivatives[leaf_idx] = leaf_derivative
                da[updated_leaf_documents_idxs] += leaf_derivative
            destination_model.influence_trees.append(
                CatBoostTree.copy_tree_with_replaced_leaf_values(new_tree, leaf_derivatives)
            )
        return destination_model

    def prediction_derivative(self, documents):
        return reduce(lambda x, y: x + y, (tree(documents) for tree in self.influence_trees))

    def loss_derivative(self, documents, targets):
        return self.trees[-1].loss_function.gradient(targets, self(documents)) * self.prediction_derivative(documents)

    def _initialize_influence_trees(self, train_documents, train_targets, loss_function, weights,
                                    l2_regularization_coef, learning_rate):
        assert all(isinstance(tree, CBLeafInfluenceObliviousTree) for tree in self.trees), type(self.trees[0])
        current_approxes = np.zeros_like(train_targets, dtype=np.float64)
        for t, tree in enumerate(self.trees):
            tree._initialize_influence_tree(train_documents, train_targets, current_approxes, loss_function, weights,
                                            l2_regularization_coef, learning_rate)
            current_approxes += tree._document_predictions_in_this_tree

    def _documents_to_update_method_from_update_set_param(self, update_set, **update_set_params):
        if update_set == 'SinglePoint':
            return self._only_removed_point_idx_set
        elif update_set == 'AllPoints':
            return self._all_points_idxs_set
        elif update_set == 'TopKLeaves':
            k = update_set_params.pop('k')
            return lambda t, idx, da: self._top_k_leaves_idxs_set(k, t, idx, da)
        else:
            raise ValueError('Unknown update set param %s' % update_set)

    def _only_removed_point_idx_set(self, t, removed_point_idx, da):
        return {removed_point_idx}

    def _all_points_idxs_set(self, t, removed_point_idx, da):
        return set(range(self._train_documents_num))

    def _top_k_leaves_idxs_set(self, k, t, removed_point_idx, da):
        tree = self.trees[t]
        leaf_das = [
            np.sum(np.abs(da[list(tree._document_idxs_for_leaves[leaf_idx])]))
            for leaf_idx in xrange(len(tree.leaf_values))
        ]
        leaf_idxs_to_update = np.argsort(leaf_das)[-k:]
        return reduce(lambda x, y: x | y, (tree._document_idxs_for_leaves[l] for l in leaf_idxs_to_update))| {removed_point_idx}

    @staticmethod
    def _tree_class_from_leaf_method_param(leaf_method):
        if leaf_method == 'Newton':
            return CBNewtonLeafInfluenceObliviousTree
        elif leaf_method == 'Gradient':
            return CBGradientLeafInfluenceObliviousTree
        else:
            raise ValueError('Unknown leaf method %s' % leaf_method)

    @property
    def _tree_class(self):
        return self._tree_class_
