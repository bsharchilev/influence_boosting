import logging
from copy import deepcopy
from os import mkdir
from os.path import isdir

import numpy as np
import tensorflow as tf
from catboost import CatBoostClassifier

from ..leaf_influence import CBLeafInfluenceEnsemble
from ...loss import BinaryCrossEntropyLoss
from ...util import export_catboost_to_json, read_json_params
from data.adult import read_train_documents_and_one_hot_targets

tf.logging.set_verbosity(logging.ERROR)
logger = logging.getLogger('__main__')
logger.setLevel(logging.INFO)


class TFGBApplier(object):
    def __init__(self, icb, train_documents, train_labels, leaf_method, weights=None):
        self.icb = icb
        if weights is None:
            self.weights_num = np.ones_like(train_labels, dtype=np.float64)
        else:
            self.weights_num = np.array(weights, dtype=tf.float64)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train_documents = train_documents
        self.train_labels = train_labels
        self.weights = tf.placeholder(tf.float64, shape=train_labels.shape, name='weights')
        self.x = tf.placeholder(tf.float64, shape=train_documents.shape, name='x')
        self.y = tf.placeholder(tf.float64, shape=train_labels.shape, name='y')
        self.approxes = []
        self.approxes.append(tf.zeros_like(train_labels, dtype=tf.float64))
        self.leaf_values = []
        self.leaf_values_grads = []
        for t in xrange(len(icb.trees)):
            a = self.approxes[-1]
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=a,
                                                                           name='loss_step_%s' % str(t)))
            grads = tf.gradients(loss, a)[0]
            hessians = tf.diag_part(tf.hessians(loss, a)[0])
            leaf_doc_idxs = [sorted(list(l)) for l in icb.trees[t]._document_idxs_for_leaves]
            doc_leaf_idxs = [0] * len(train_labels)
            for l, leaf_idxs in enumerate(leaf_doc_idxs):
                for i in leaf_idxs:
                    doc_leaf_idxs[i] = l
            doc_leaf_idxs = tf.constant(doc_leaf_idxs, dtype=tf.int32)
            leaf_values_lst = []
            for l in xrange(len(icb.trees[t].leaf_values)):
                leaf_mask = tf.equal(doc_leaf_idxs, l)
                leaf_gradients = tf.boolean_mask(grads, leaf_mask)
                leaf_hessians = tf.boolean_mask(hessians, leaf_mask)
                leaf_weights = tf.boolean_mask(self.weights, leaf_mask)
                if leaf_method == 'Gradient':
                    leaf_values_lst.append(-tf.divide(
                        tf.reduce_sum(tf.multiply(leaf_weights, leaf_gradients)),
                        tf.reduce_sum(leaf_weights) + icb.trees[t].l2_reg_coef
                    ) * icb.trees[t].learning_rate)
                else:
                    leaf_values_lst.append(-tf.divide(
                        tf.reduce_sum(tf.multiply(leaf_weights, leaf_gradients)),
                        tf.reduce_sum(tf.multiply(leaf_weights, leaf_hessians)) + icb.trees[t].l2_reg_coef
                    ) * icb.trees[t].learning_rate)
            leaf_values = tf.stack(leaf_values_lst)
            self.leaf_values.append(leaf_values)
            tree_predictions = tf.gather(leaf_values, doc_leaf_idxs)
            self.approxes.append(a + tree_predictions)
            leaf_value_grad = []
            for lv in leaf_values_lst:
                lvg = tf.gradients(lv, self.weights)[0]
                leaf_value_grad.append(lvg)
            self.leaf_values_grads.append(leaf_value_grad)
        self.train_idx = tf.placeholder(tf.int32, shape=[])
        train_prediction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y[self.train_idx:self.train_idx + 1], logits=self.approxes[-1][self.train_idx:self.train_idx + 1]
        ))
        self.train_prediction_loss_grad = tf.gradients(train_prediction_loss, self.weights)[0]

    def get_predicts(self):
        return self.sess.run(self.approxes[-1], feed_dict={
            self.weights: self.weights_num, self.x: self.train_documents, self.y: self.train_labels
        })

    def get_derivs(self, i):
        lvgs = self.sess.run(self.leaf_values_grads, feed_dict={
            self.weights: self.weights_num, self.x: self.train_documents, self.y: self.train_labels
        })  # [num_trees, num_leaves, num_train_objs]
        return [[lvs[i] for lvs in t] for t in lvgs]

    def get_train_prediction_deriv(self, i_removed, i_train):
        res = self.sess.run(self.train_prediction_loss_grad, feed_dict={
            self.weights: self.weights_num, self.x: self.train_documents, self.y: self.train_labels, self.train_idx: i_train
        })
        return res[i_removed]


def _test_influence_vs_tf_derivative(leaf_method):
    base_dir = 'data/adult/'
    train_documents, train_targets = read_train_documents_and_one_hot_targets(
        base_dir + 'train_data_catboost_format.tsv'
    )
    train_documents = train_documents[:100]
    train_targets = train_targets[:100]

    train_targets = np.argmax(train_targets, axis=1)

    test_documents, test_targets = read_train_documents_and_one_hot_targets(
        base_dir + 'test_data_catboost_format.tsv'
    )
    test_targets = np.argmax(test_targets, axis=1)

    train_dir = base_dir + 'ut_tmp/'
    if not isdir(train_dir):
        mkdir(train_dir)
    cbc_params = read_json_params(base_dir + 'catboost_params.json')
    cbc_params['iterations'] = 2
    cbc_params['leaf_estimation_method'] = leaf_method
    cbc_params['random_seed'] = 10
    cbc_params['train_dir'] = train_dir
    cbc = CatBoostClassifier(**cbc_params)
    cbc.set_params(boosting_type='Plain')
    cbc.fit(train_documents, train_targets)
    cbc.save_model(train_dir + 'model.bin', format='cbm')
    export_catboost_to_json(train_dir + 'model.bin', train_dir + 'model.json')
    full_model = CBLeafInfluenceEnsemble(train_dir + 'model.json', train_documents, train_targets,
                                         leaf_method=leaf_method,
                                         learning_rate=cbc_params['learning_rate'],
                                         loss_function=BinaryCrossEntropyLoss(),
                                         update_set='AllPoints')
    retrained_model_our = deepcopy(full_model)
    tf_checker = TFGBApplier(full_model, train_documents, train_targets, leaf_method)
    for remove_idx in np.random.randint(len(train_targets), size=30):
        full_model.fit(remove_idx, retrained_model_our)
        pred_ours = full_model(train_documents)
        pred_theirs = tf_checker.get_predicts()
        pred_cbc = cbc.predict(train_documents, prediction_type='RawFormulaVal')
        assert np.allclose(pred_ours, pred_theirs, rtol=1e-3) and np.allclose(pred_ours, pred_cbc, rtol=1e-3), (pred_ours, pred_theirs)

        der_ours = [t.leaf_values for t in retrained_model_our.influence_trees]
        der_theirs = tf_checker.get_derivs(remove_idx)
        assert all(np.allclose(o, t, rtol=1e-2) for o, t in zip(der_ours, der_theirs)), (der_ours, der_theirs)

        random_train_idx = np.random.randint(len(train_targets))
        der_pred_ours = retrained_model_our.loss_derivative(train_documents[[random_train_idx]],
                                                            train_targets[[random_train_idx]])[0]
        der_pred_theirs = tf_checker.get_train_prediction_deriv(remove_idx, random_train_idx)
        assert np.isclose(der_pred_ours, der_pred_theirs, rtol=1e-2), (der_pred_ours, der_pred_theirs)


def _test_prediction_consistency(leaf_method):
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
    cbc_params['leaf_estimation_method'] = leaf_method
    cbc_params['random_seed'] = 10
    cbc_params['train_dir'] = train_dir
    cbc = CatBoostClassifier(**cbc_params)
    cbc.set_params(boosting_type='Plain')
    cbc.fit(train_documents, train_targets)
    cbc.save_model(train_dir + 'model.bin', format='cbm')
    export_catboost_to_json(train_dir + 'model.bin', train_dir + 'model.json')
    full_model = CBLeafInfluenceEnsemble(train_dir + 'model.json', train_documents, train_targets,
                                         learning_rate=cbc_params['learning_rate'],
                                         loss_function=BinaryCrossEntropyLoss(),
                                         leaf_method=leaf_method,
                                         update_set='AllPoints')
    assert np.allclose(full_model(train_documents), cbc.predict(train_documents, prediction_type='RawFormulaVal'), rtol=0.001),\
        [(a,b)
         for a, b in zip(full_model(train_documents), cbc.predict(train_documents, prediction_type='RawFormulaVal'))
         if not np.allclose(a, b)]
    assert np.allclose(full_model(test_documents), cbc.predict(test_documents, prediction_type='RawFormulaVal'), rtol=0.001)


def test_influence_vs_tf_derivative_gradient():
    return _test_influence_vs_tf_derivative('Gradient')


def test_influence_vs_tf_derivative_newton():
    return _test_influence_vs_tf_derivative('Newton')


def test_prediction_consistency_gradient():
    return _test_prediction_consistency('Gradient')


def test_prediction_consistency_newton():
    return _test_prediction_consistency('Newton')


if __name__ == '__main__':
    test_influence_vs_tf_derivative_gradient()
    test_influence_vs_tf_derivative_newton()
    test_prediction_consistency_gradient()
    test_prediction_consistency_newton()