from .utilities import _flip_patches_no_diag
import tensorflow as tf


# Taylor approximation idea taken from Smith: Leveraging Synthetic Images with Domain-Adversarial
# Neural Networks for Fine-Grained Car Model Classification,
# https://urn.kb.se/resolve?urn=urn%3Anbn%3Ase%3Akth%3Adiva-307616
def _log_approx(x, order=2):

    if order < 2:
        raise ValueError("Approx. requires order of at least 2. For order 1, consider Brier score.")

    ret = tf.subtract(x, tf.constant(1.0))

    for i in range(2, order + 1):

        mult_factor = tf.constant(1 / i, dtype=x.dtype.base_dtype)
        pow_factor = tf.constant(i, dtype=x.dtype.base_dtype)
        if (i % 2) == 0:  # even
            appl_func = tf.subtract
        else:
            appl_func = tf.add

        ret = appl_func(ret, tf.multiply(mult_factor, tf.pow(tf.subtract(x, tf.constant(1.0)), pow_factor)))

    return ret


def sec_approx_categ_ce(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):

    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, y_pred.dtype)

    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / tf.reduce_sum(y_pred, axis, True)

    # Compute cross entropy from probabilities.
    epsilon_ = tf.constant(1e-07, dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
    y_pred_log = _log_approx(y_pred, order=100)

    return -tf.reduce_sum(y_true * y_pred_log, axis)


def sec_approx_binary_ce(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):

    epsilon_ = tf.constant(1e-07, dtype=y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Compute cross entropy from probabilities.
    bce = y_true * _log_approx(y_pred + epsilon_, order=100)
    bce += (1 - y_true) * _log_approx(1 - y_pred + epsilon_, order=100)
    return -bce


# Taken from: A Simple Framework for Contrastive Learning of Visual Representations, Chen et al.
# https://proceedings.mlr.press/v119/chen20j.html
@tf.function
def nt_xent_loss(pij, tau):
    batch_size_times_two = tf.shape(pij)[0]

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_projections, right_projections = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=pij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=pij)

    similarity_matrix = -1 \
        * tf.keras.losses.cosine_similarity(right_projections, left_projections, axis=-1) \
        * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    nominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    logit_1 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_1, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_1, tau)), axis=-1)
    ))

    nominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.range(1, batch_size_times_two, delta=2),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    logit_2 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_2, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_2, tau)), axis=-1)
    ))

    return tf.reduce_sum(logit_1 + logit_2) / tf.cast(batch_size_times_two, dtype=tf.float32)


# Taken from: Decoupled Contrastive Learning, Yeh et al.
# https://link.springer.com/chapter/10.1007/978-3-031-19809-0_38
@tf.function
def decoupl_nt_xent_loss(pij, tau):
    batch_size_times_two = tf.shape(pij)[0]

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_projections, right_projections = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=pij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=pij)

    similarity_matrix = -1 \
        * tf.keras.losses.cosine_similarity(right_projections, left_projections, axis=-1) \
        * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    nominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    denominator_1_upd = tf.scatter_nd(
        updates=nominator_1,
        shape=(tf.shape(denominator_1)[0], tf.shape(denominator_1)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1_decoupl = tf.subtract(denominator_1, denominator_1_upd)
    logit_1 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_1, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_1_decoupl, tau)), axis=-1)
    ))

    nominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.range(1, batch_size_times_two, delta=2),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    denominator_2_upd = tf.scatter_nd(
        updates=nominator_2,
        shape=(tf.shape(denominator_2)[0], tf.shape(denominator_2)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2_decoupl = tf.subtract(denominator_2, denominator_2_upd)
    logit_2 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_2, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_2_decoupl, tau)), axis=-1)
    ))

    return tf.reduce_sum(logit_1 + logit_2) / tf.cast(batch_size_times_two, dtype=tf.float32)


@tf.function
def dom_aware_decoupl_nt_xent_loss(pij, tau):
    batch_size_times_two = tf.shape(pij)[0]

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    right_indices = tf.squeeze(_flip_patches_no_diag(right_indices[tf.newaxis, :, :, tf.newaxis], size_x=2, size_y=2))
    left_projections, right_projections = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=pij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=pij)

    similarity_matrix = -1 \
        * tf.keras.losses.cosine_similarity(right_projections, left_projections, axis=-1) \
        * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    nominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    denominator_1_upd = tf.scatter_nd(
        updates=nominator_1,
        shape=(tf.shape(denominator_1)[0], tf.shape(denominator_1)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1_decoupl = tf.subtract(denominator_1, denominator_1_upd)
    logit_1 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_1, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_1_decoupl, tau)), axis=-1)
    ))

    nominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.range(1, batch_size_times_two, delta=2),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    denominator_2_upd = tf.scatter_nd(
        updates=nominator_2,
        shape=(tf.shape(denominator_2)[0], tf.shape(denominator_2)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2_decoupl = tf.subtract(denominator_2, denominator_2_upd)
    logit_2 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_2, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_2_decoupl, tau)), axis=-1)
    ))

    return tf.reduce_sum(logit_1 + logit_2) / tf.cast(batch_size_times_two, dtype=tf.float32)


@tf.function
def dom_aware_decoupl_nt_xent_loss_2(pij, sij, tau):
    batch_size_times_two = tf.shape(pij)[0]

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_projections, right_projections = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=pij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=pij)
    left_labels, right_labels = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=sij), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=sij)

    similarity_matrix = -1 \
        * tf.keras.losses.cosine_similarity(right_projections, left_projections, axis=-1) \
        * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)
    label_mask = tf.cast(tf.equal(
        tf.argmax(left_labels, axis=-1), tf.argmax(right_labels, axis=-1)
    ), dtype=tf.float32)

    nominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    label_mask_denominator_1 = tf.gather_nd(
        params=label_mask,
        indices=tf.expand_dims(tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1), axis=-1)
    )
    denominator_1_upd = tf.scatter_nd(
        updates=nominator_1,
        shape=(tf.shape(denominator_1)[0], tf.shape(denominator_1)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.range(1, batch_size_times_two, delta=2)], axis=1)
    )
    denominator_1_decoupl = tf.subtract(denominator_1, denominator_1_upd)
    denominator_1_decoupl = tf.multiply(denominator_1_decoupl, label_mask_denominator_1)
    logit_1 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_1, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_1_decoupl, tau)), axis=-1)
    ))

    nominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.stack([tf.range(1, batch_size_times_two, delta=2),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2 = tf.gather_nd(
        params=similarity_matrix,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    label_mask_denominator_2 = tf.gather_nd(
        params=label_mask,
        indices=tf.expand_dims(tf.range(1, batch_size_times_two, delta=2), axis=-1)
    )
    denominator_2_upd = tf.scatter_nd(
        updates=nominator_2,
        shape=(tf.shape(denominator_2)[0], tf.shape(denominator_2)[-1]),
        indices=tf.stack([tf.range(0, batch_size_times_two // 2, delta=1),
                          tf.subtract(tf.range(1, batch_size_times_two, delta=2), 1)], axis=1)
    )
    denominator_2_decoupl = tf.subtract(denominator_2, denominator_2_upd)
    denominator_2_decoupl = tf.multiply(denominator_2_decoupl, label_mask_denominator_2)
    logit_2 = -tf.math.log(tf.math.divide(
        tf.exp(tf.divide(nominator_2, tau)),
        tf.reduce_sum(tf.exp(tf.divide(denominator_2_decoupl, tau)), axis=-1)
    ))

    return tf.reduce_sum(logit_1 + logit_2) / tf.cast(batch_size_times_two, dtype=tf.float32)


@tf.function
def ml_xent_loss(y_true, y_logits):
    batch_size_times_two = tf.shape(y_logits)[0]
    y_true = tf.repeat(tf.cast(y_true, dtype=tf.int16), repeats=[2] * tf.shape(y_true)[0], axis=0)

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_logits, right_logits = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=y_logits), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=y_logits)
    left_labels, right_labels = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=y_true), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=y_true)

    logits = tf.math.sigmoid(tf.add(left_logits, right_logits))
    labels = tf.clip_by_value(tf.add(left_labels, right_labels), clip_value_min=0, clip_value_max=1)

    loss_matrix = tf.math.reduce_mean(sec_approx_binary_ce(labels, logits), axis=-1) \
                  * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    overall_loss = tf.reduce_sum(loss_matrix) / tf.pow(tf.cast(batch_size_times_two, dtype=tf.float32), tf.constant(2.0, dtype=tf.float32))

    return overall_loss


@tf.function
def ml_xent_loss_2(y_true, y_logits):
    batch_size_times_two = tf.shape(y_logits)[0]
    y_true = tf.repeat(tf.cast(y_true, dtype=tf.int16), repeats=[2] * tf.shape(y_true)[0], axis=0)

    left_indices, right_indices = tf.meshgrid(tf.range(batch_size_times_two), tf.range(batch_size_times_two))
    left_logits, right_logits = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=y_logits), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=y_logits)
    left_labels, right_labels = \
        tf.gather_nd(indices=tf.expand_dims(left_indices, axis=-1), params=y_true), \
        tf.gather_nd(indices=tf.expand_dims(right_indices, axis=-1), params=y_true)

    if (tf.shape(y_true)[-1] % 2) == 0:  # even
        logits = tf.math.sigmoid(tf.concat([left_logits, right_logits], axis=-1))
    else:
        logits = tf.math.sigmoid(tf.concat([left_logits, right_logits], axis=-1)[..., :-1])

    labels = tf.clip_by_value(tf.add(left_labels, right_labels), clip_value_min=0, clip_value_max=1)

    loss_matrix = tf.math.reduce_mean(sec_approx_binary_ce(labels, logits), axis=-1) \
                  * tf.cast(~tf.eye(batch_size_times_two, dtype=tf.bool), tf.float32)

    overall_loss = tf.reduce_sum(loss_matrix) / tf.pow(tf.cast(batch_size_times_two, dtype=tf.float32), tf.constant(2.0, dtype=tf.float32))

    return overall_loss
