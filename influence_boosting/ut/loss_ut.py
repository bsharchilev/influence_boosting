import numpy as np
import tensorflow as tf

from ..loss import CrossEntropyLoss, BinaryCrossEntropyLoss


def _test_loss(sample_shape_fn):
    our_loss_fn = CrossEntropyLoss()

    single_vector_inputs = len(sample_shape_fn()) == 1 and sample_shape_fn() == sample_shape_fn()
    length = [None] if not single_vector_inputs else [sample_shape_fn()[0]]

    targets_ph = tf.placeholder(tf.float64, length * len(sample_shape_fn()))
    logits_ph = tf.placeholder(tf.float64, length * len(sample_shape_fn()))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets_ph, logits=logits_ph)
    grad = tf.gradients(loss, logits_ph)
    if single_vector_inputs:
        ihvp = tf.reshape(tf.matrix_solve(tf.hessians(loss, logits_ph)[0] + tf.eye(length[0], dtype=tf.float64),
                                          tf.reshape(grad, (-1,1))), (-1,))
    session = tf.Session()
    for _ in xrange(1000):
        shape = sample_shape_fn()
        targets = np.random.rand(*shape)
        targets = targets ** 2
        targets /= np.sum(targets, axis=-1, keepdims=True)
        logits = np.random.rand(*shape)

        our_loss = our_loss_fn(targets, logits)
        our_grad = our_loss_fn.gradient(targets, logits)
        if single_vector_inputs:
            our_ihvp = our_loss_fn.ihvp(targets, logits, l2_reg=1)

        feed_dict = {targets_ph: targets, logits_ph: logits}
        true_loss = session.run(loss, feed_dict=feed_dict)
        true_gradient = session.run(grad, feed_dict=feed_dict)
        if single_vector_inputs:
            true_ihvp = session.run(ihvp, feed_dict=feed_dict)

        assert np.allclose(our_loss, true_loss) and np.allclose(our_grad, true_gradient)
        if single_vector_inputs:
            assert np.allclose(our_ihvp, true_ihvp)

def test_batch_vector_loss():
    def generate_2d_array_shape():
        return np.random.randint(2, 1000), np.random.randint(2, 20)
    _test_loss(generate_2d_array_shape)


def test_single_vector_loss():
    def generate_1d_array_shape():
        return (np.random.randint(2, 20),)
    _test_loss(generate_1d_array_shape)


def test_constant_length_single_vector():
    # TF can compute hessians only with constant-shape inputs
    length = np.random.randint(1, 10)
    def fixed_1d_array_shape():
        return (length,)
    _test_loss(fixed_1d_array_shape)


def test_log_loss():
    our_loss_fn = BinaryCrossEntropyLoss()
    target_ph = tf.placeholder(tf.float64, (100,))
    logits_ph = tf.placeholder(tf.float64, (100,))
    loss_vector = -target_ph * tf.log(tf.sigmoid(logits_ph)) - (1 - target_ph) * tf.log(1 - tf.sigmoid(logits_ph))
    loss = tf.reduce_sum(loss_vector)
    grad = tf.gradients(loss, logits_ph)
    seconders = tf.diag_part(tf.hessians(loss, logits_ph)[0])
    s = tf.Session()
    for _ in xrange(100):
        random_targets = (np.random.rand(100) > 0.5).astype(int)
        random_logits = np.random.rand(100) * 2
        fd = {target_ph: random_targets, logits_ph: random_logits}
        true_losses = s.run(loss_vector, fd)
        true_grad = s.run(grad, fd)
        true_seconders = s.run(seconders, fd)

        our_losses = our_loss_fn(random_targets, random_logits)
        our_grad = our_loss_fn.gradient(random_targets, random_logits)
        our_seconders = our_loss_fn.hessian(random_targets, random_logits)

        assert np.allclose(true_losses, our_losses), (true_losses, our_losses)
        assert np.allclose(true_grad, our_grad)
        assert np.allclose(true_seconders, our_seconders), (true_seconders, our_seconders)

if __name__ == '__main__':
    test_log_loss()
    test_batch_vector_loss()
    test_single_vector_loss()
    test_constant_length_single_vector()
