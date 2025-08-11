import tensorflow_addons as tfa
import tensorflow as tf


# Sample mixing, gaussian noise augmenter methods are taken from
# 'Data Augmentation and Dense-LSTM for Human Activity Recognition Using WiFi Signal':
# https://doi.org/10.1109/JIOT.2020.3026732

# Phase and magnitude scaling augmentation methods are taken from
# 'Simple and Effective Augmentation Methods for CSI Based Indoor Localization':
# https://arxiv.org/abs/2211.10790

# Warping augmentation is taken from
# 'Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks':
# https://doi.org/10.1145/3136755.3136817
# Code link: https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb


# 1D np.interp equivalent taken from: https://brentspell.com/2022/tensorflow-interp/
def tf_interp(x, xs, ys):
    # determine the output data type
    ys = tf.convert_to_tensor(ys)
    dtype = ys.dtype

    # normalize data types
    ys = tf.cast(ys, tf.float64)
    xs = tf.cast(xs, tf.float64)
    x = tf.cast(x, tf.float64)

    # pad control points for extrapolation
    xs = tf.concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    # compute slopes, pad at the edges to flatten
    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    # solve for intercepts
    bs = ys - ms * xs

    # search for the line parameters at each input data point
    # create a grid of the inputs and piece breakpoints for thresholding
    # rely on argmax stopping on the first true when there are duplicates,
    # which gives us an index into the parameter vectors
    i = tf.math.argmax(xs[..., tf.newaxis, :] > x[..., tf.newaxis], axis=-1)
    m = tf.gather(ms, i, axis=-1)
    b = tf.gather(bs, i, axis=-1)

    # apply the linear mapping at each input data point
    y = m * x + b
    return tf.cast(tf.reshape(y, tf.shape(x)), dtype)


def gen_rand_curves(x, sigma, knot):

    orig_shape = x.shape

    if len(orig_shape) == 4:
        x = tf.reshape(tf.transpose(x, perm=[1, 0, 2, 3]), shape=(orig_shape[1], -1, orig_shape[-1]))

    xx = tf.ones(shape=(x.shape[1] * x.shape[2], 1)) * \
          tf.range(0, x.shape[0], (x.shape[0] - 1) / (knot + 1), dtype=tf.float32)
    xx = tf.expand_dims(xx, axis=-1)
    yy = tf.random.normal(shape=(x.shape[1] * x.shape[2], knot + 2), mean=1.0, stddev=sigma)
    yy = tf.expand_dims(yy, axis=-1)
    x_range = tf.range(x.shape[0], dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
    x_range = tf.broadcast_to(x_range, shape=(xx.shape[0], x_range.shape[1], x_range.shape[2]))

    to_return = tf.reshape(tf.transpose(tf.squeeze(tfa.image.interpolate_spline(train_points=xx, train_values=yy, query_points=x_range, order=3, regularization_weight=0.0))), shape=x.shape)

    if len(orig_shape) == 4:
        to_return = tf.reshape(to_return, shape=(to_return.shape[0], orig_shape[0], -1, to_return.shape[-1]))
        to_return = tf.transpose(to_return, perm=[1, 0, 2, 3])

    return to_return


def distort_time_steps(x, sigma, knot):
    tt = gen_rand_curves(x=x, sigma=sigma, knot=knot)

    if len(x.shape) == 4:
        tt_cum = tf.math.cumsum(tt, axis=1)
        last_elem = tt_cum[:, -1, :, :]
        time_dim_value = x.shape[1] - 1
        t_scale = tf.divide(time_dim_value, last_elem)[:, tf.newaxis, :, :]
    else:
        tt_cum = tf.math.cumsum(tt, axis=0)
        last_elem = tt_cum[-1, :, :]
        time_dim_value = x.shape[0] - 1
        t_scale = tf.divide(time_dim_value, last_elem)[tf.newaxis, :, :]

    return tf.multiply(tt_cum, t_scale)


def time_warp(x, sigma, knot):

    orig_shape = x.shape
    tt_new = distort_time_steps(x=x, sigma=sigma, knot=knot)

    if len(x.shape) == 4:
        x = tf.reshape(tf.transpose(x, perm=[1, 0, 2, 3]), shape=(orig_shape[1], -1))
        tt_new = tf.reshape(tf.transpose(tt_new, perm=[1, 0, 2, 3]), shape=(orig_shape[1], -1))
    else:
        x = tf.reshape(x, shape=(orig_shape[0], -1))
        tt_new = tf.reshape(tt_new, shape=(orig_shape[0], -1))

    x_range = tf.broadcast_to(tf.range(x.shape[0], dtype=tf.float32)[:, tf.newaxis], shape=x.shape)
    x_range = tf.transpose(x_range)
    x = tf.transpose(x)
    tt_new = tf.transpose(tt_new)

    interp_result = tf.map_fn(fn=lambda elem: tf_interp(elem[0], elem[1], elem[2]), elems=(x_range, tt_new, x),
                              fn_output_signature=tf.TensorSpec(shape=(x.shape[-1],), dtype=tf.float32))

    if len(orig_shape) == 4:
        return tf.transpose(tf.reshape(tf.transpose(interp_result), shape=(orig_shape[1], orig_shape[0], orig_shape[2], orig_shape[3])), perm=[1, 0, 2, 3])
    else:
        return tf.reshape(tf.transpose(interp_result), shape=(orig_shape[0], orig_shape[1], orig_shape[2]))


def time_permute(x, window_size):

    if len(x.shape) == 4:
        windowed_shape = (tf.shape(x)[0], tf.shape(x)[1] // window_size, window_size, tf.shape(x)[2], tf.shape(x)[3])
        roll_axis = 1
        perm_indices = [1, 0, 2, 3, 4]
    else:
        windowed_shape = (tf.shape(x)[0] // window_size, window_size, tf.shape(x)[1], tf.shape(x)[2])
        roll_axis = 0
        perm_indices = [0, 1, 2, 3]

    x_reshape = tf.reshape(x, shape=windowed_shape)
    x_reshape2 = tf.roll(x_reshape, shift=1, axis=roll_axis)

    gather_indices = tf.range(start=0, limit=tf.shape(x_reshape)[roll_axis], delta=2)
    stitch_indices = tf.split(tf.random.shuffle(tf.range(start=0, limit=tf.shape(x_reshape)[roll_axis], delta=1)), num_or_size_splits=2)

    gather_result = tf.transpose(tf.gather(x_reshape, indices=gather_indices, axis=roll_axis), perm=perm_indices)
    gather_result2 = tf.transpose(tf.gather(x_reshape2, indices=gather_indices, axis=roll_axis), perm=perm_indices)

    return tf.reshape(tf.transpose(tf.dynamic_stitch(indices=stitch_indices, data=[gather_result2, gather_result]), perm=perm_indices), shape=x.shape)


def phase_warp(x, sigma, knot):
    if len(x.shape) == 4:
        new_compl_vals = tf.cast(x[:, :, :x.shape[2] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, :, x.shape[2] // 2:, :], dtype=tf.complex64))
        warp_matrix = gen_rand_curves(new_compl_vals, sigma=sigma, knot=knot)
        rate = tf.math.exp(1j * tf.cast(warp_matrix, dtype=tf.complex64))
        new_compl_vals = new_compl_vals * rate

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=2)

    else:
        new_compl_vals = tf.cast(x[:, :x.shape[1] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, x.shape[1] // 2:, :], dtype=tf.complex64))
        warp_matrix = gen_rand_curves(new_compl_vals, sigma=sigma, knot=knot)
        rate = tf.math.exp(1j * tf.cast(warp_matrix, dtype=tf.complex64))
        new_compl_vals = new_compl_vals * rate

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=1)


def phase_scale(x, rate):
    if len(x.shape) == 4:
        new_compl_vals = tf.cast(x[:, :, :x.shape[2] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, :, x.shape[2] // 2:, :], dtype=tf.complex64))
        rate = tf.broadcast_to(rate[tf.newaxis, tf.newaxis, tf.newaxis, :], shape=new_compl_vals.shape)
        rate = tf.math.exp(1j * tf.cast(rate, dtype=tf.complex64))
        new_compl_vals = new_compl_vals * rate

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=2)

    else:
        new_compl_vals = tf.cast(x[:, :x.shape[1] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, x.shape[1] // 2:, :], dtype=tf.complex64))
        rate = tf.broadcast_to(rate[tf.newaxis, tf.newaxis, :], shape=new_compl_vals.shape)
        rate = tf.math.exp(1j * tf.cast(rate, dtype=tf.complex64))
        new_compl_vals = new_compl_vals * rate

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=1)


def magn_warp(x, sigma, knot):
    if len(x.shape) == 4:
        new_compl_vals = tf.cast(x[:, :, :x.shape[2] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, :, x.shape[2] // 2:, :], dtype=tf.complex64))
        warp_matrix = gen_rand_curves(new_compl_vals, sigma=sigma, knot=knot)
        new_compl_vals = new_compl_vals * tf.cast(warp_matrix, dtype=tf.complex64)

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=2)

    else:
        new_compl_vals = tf.cast(x[:, :x.shape[1] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, x.shape[1] // 2:, :], dtype=tf.complex64))
        warp_matrix = gen_rand_curves(new_compl_vals, sigma=sigma, knot=knot)
        new_compl_vals = new_compl_vals * tf.cast(warp_matrix, dtype=tf.complex64)

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=1)


def magn_scale(x, rate):
    if len(x.shape) == 4:
        new_compl_vals = tf.cast(x[:, :, :x.shape[2] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, :, x.shape[2] // 2:, :], dtype=tf.complex64))
        rate = tf.broadcast_to(rate[tf.newaxis, tf.newaxis, tf.newaxis, :], shape=new_compl_vals.shape)
        new_compl_vals = new_compl_vals * tf.cast(rate, dtype=tf.complex64)

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=2)

    else:
        new_compl_vals = tf.cast(x[:, :x.shape[1] // 2, :], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(x[:, x.shape[1] // 2:, :], dtype=tf.complex64))
        rate = tf.broadcast_to(rate[tf.newaxis, tf.newaxis, :], shape=new_compl_vals.shape)
        new_compl_vals = new_compl_vals * tf.cast(rate, dtype=tf.complex64)

        return tf.concat([tf.abs(new_compl_vals), tf.math.angle(new_compl_vals)], axis=1)


def sample_mixing(x, rate):
    x = tf.stack(x, axis=0)

    rate = tf.convert_to_tensor(rate, dtype=tf.float32)
    rate_update = tf.subtract(tf.constant(1., dtype=tf.float32), tf.gather(rate, [0]))
    rate = tf.tensor_scatter_nd_update(rate, tf.constant([[0]]), rate_update)

    if len(x.shape) == 5:
        rate = rate[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    else:
        rate = rate[:, tf.newaxis, tf.newaxis, tf.newaxis]

    rate = tf.broadcast_to(rate, x.shape)
    ret = tf.multiply(x, rate)

    return tf.reduce_sum(ret, axis=0)


def augment(x, data_format, dataset):

    if len(x[0].shape) == 4:
        time_window_shape = x[0].shape[1]
    else:
        time_window_shape = x[0].shape[0]

    func_selection_nr = tf.random.uniform(shape=(), minval=0, maxval=3, seed=42, dtype=tf.int32)
    augmented_sample = tf.switch_case(branch_index=func_selection_nr, branch_fns={
        0: lambda: x[0],
        1: lambda: tf.keras.layers.GaussianNoise(tf.random.uniform(shape=(), minval=0.0, maxval=0.11, dtype=tf.float32))(x[0], training=True),
        2: lambda: time_permute(x[0], window_size=time_window_shape // 8),
        3: lambda: sample_mixing(x, tf.random.uniform(shape=(3,), minval=0.0, maxval=0.05, dtype=tf.float32))
    }, default=lambda: x[0])

    if data_format == 'ampphase' and dataset == 'widar3':
        return tf.image.pad_to_bounding_box(augmented_sample, 24, 2, 2048, 64)
    elif data_format == 'ampphase' and dataset == 'signfi':
        return tf.image.pad_to_bounding_box(augmented_sample, 12, 2, 224, 64)
    else:
        raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")


def no_augment(x, data_format, dataset):
    if data_format == 'ampphase' and dataset == 'widar3':
        return tf.image.pad_to_bounding_box(x[0], 24, 2, 2048, 64)
    elif data_format == 'ampphase' and dataset == 'signfi':
        return tf.image.pad_to_bounding_box(x[0], 12, 2, 224, 64)
    else:
        raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")
