from tensorflow_probability.python.internal import prefer_static as ps
import tensorflow as tf


# https://github.com/tensorflow/tensorflow/issues/36327
def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(layer, batch_size=batch_size)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

    total_memory = (batch_size * shapes_mem_count + internal_model_mem_count + trainable_count + non_trainable_count)
    return round(total_memory * 1.15)  # To account for 10% discrepancy as indicated by author


# Assumes row-major flattening procedure for creation of domain label
# (see: https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays)
def domain_to_be_left_out_indices_calculation(dim_nr, dim_ind, unflat_domain_label_shape):
    base_dim_range_lists = [list(range(x)) for x in unflat_domain_label_shape]
    base_dim_range_lists[dim_nr] = [dim_ind]

    test_indices = [
        w + unflat_domain_label_shape[-1] * x + unflat_domain_label_shape[-1] * unflat_domain_label_shape[-2] * y +
        unflat_domain_label_shape[-1] * unflat_domain_label_shape[-2] * unflat_domain_label_shape[-3] * z for w in
        base_dim_range_lists[-1] for x in base_dim_range_lists[-2] for y in base_dim_range_lists[-3] for z in
        base_dim_range_lists[-4]]

    # test_indices.sort()
    # print(test_indices)

    return test_indices


def gaf(x, paa_window_size=1, method='gasf', interpolate=False):
    tf.debugging.assert_type(x, tf_type=tf.float32, message="Input values must be of float32 type.")
    tf.debugging.assert_rank(x, 2, message="For GASF to work, input tensor rank must be 2.")
    tf.debugging.assert_all_finite(x, message="NaN/Inf values encountered in input matrix.")

    n_timestamps = tf.shape(x)[0]

    if not interpolate:
        x_paa = _windowed_mean(x=x, axis=0,
                    low_indices=tf.range(start=0, limit=n_timestamps - paa_window_size + 1, delta=paa_window_size),
                    high_indices=tf.range(start=paa_window_size, limit=n_timestamps + 1, delta=paa_window_size))
    else:
        x_paa = x

    x_cos = _min_max_scale(x_paa, 1.0, -1.0)
    x_sin = tf.sqrt(tf.clip_by_value(tf.subtract(1.0, tf.pow(x_cos, 2)), clip_value_min=0.0, clip_value_max=1.0))

    if method == 'gasf':
        gasf = tf.subtract(
            tf.multiply(tf.expand_dims(x_cos, axis=1), tf.expand_dims(x_cos, axis=0)),
            tf.multiply(tf.expand_dims(x_sin, axis=1), tf.expand_dims(x_sin, axis=0))
        )

        if interpolate:
            return tf.image.resize(images=gasf[tf.newaxis, :, :, tf.newaxis],
                                   size=[tf.shape(gasf)[0] / paa_window_size, tf.shape(gasf)[1] / paa_window_size],
                                   method=tf.image.ResizeMethod.BILINEAR)
        else:
            return gasf

    elif method == 'gadf':
        gadf = tf.subtract(
            tf.multiply(tf.expand_dims(x_sin, axis=1), tf.expand_dims(x_cos, axis=0)),
            tf.multiply(tf.expand_dims(x_cos, axis=1), tf.expand_dims(x_sin, axis=0))
        )

        if interpolate:
            return tf.image.resize(images=gadf[tf.newaxis, :, :, tf.newaxis],
                                   size=[tf.shape(gadf)[0] / paa_window_size, tf.shape(gadf)[1] / paa_window_size],
                                   method=tf.image.ResizeMethod.BILINEAR)
        else:
            return gadf

    else:
        raise ValueError("Unknown GAF method. Allowed values: gasf, gadf.")


def _min_max_scale(x, target_max, target_min):
    norm_values = tf.math.divide_no_nan(
       tf.subtract(x, tf.reduce_min(x)),
       tf.subtract(tf.reduce_max(x), tf.reduce_min(x))
    )
    scaled_values = tf.add(
       tf.multiply(norm_values, tf.subtract(target_max, target_min)),
       target_min
    )
    return scaled_values


def _windowed_mean(x, low_indices=None, high_indices=None, axis=0, name=None):
    with tf.name_scope(name or 'windowed_mean'):
        x = tf.convert_to_tensor(x)
        low_indices, high_indices, low_counts, high_counts = _prepare_window_args(
            x, low_indices, high_indices, axis)

        raw_cumsum = tf.cumsum(x, axis=axis)
        cum_sums = tf.concat(
            [tf.zeros_like(tf.gather(raw_cumsum, [0], axis=axis)), raw_cumsum],
            axis=axis)
        low_sums = tf.gather(cum_sums, low_indices, axis=axis)
        high_sums = tf.gather(cum_sums, high_indices, axis=axis)

        counts = high_counts - low_counts
        return _safe_average(high_sums - low_sums, counts)


def _prepare_window_args(x, low_indices=None, high_indices=None, axis=0):
    if high_indices is None:
        high_indices = tf.range(ps.shape(x)[axis]) + 1
    else:
        high_indices = tf.convert_to_tensor(high_indices)
    if low_indices is None:
        low_indices = high_indices // 2
    else:
        low_indices = tf.convert_to_tensor(low_indices)

    # Broadcast indices together.
    high_indices = high_indices + tf.zeros_like(low_indices)
    low_indices = low_indices + tf.zeros_like(high_indices)

    size = ps.size(high_indices)
    counts_shp = ps.one_hot(
      axis, depth=ps.rank(x), on_value=size, off_value=1)

    low_counts = tf.reshape(tf.cast(low_indices, dtype=x.dtype),
                          shape=counts_shp)
    high_counts = tf.reshape(tf.cast(high_indices, dtype=x.dtype),
                           shape=counts_shp)
    return low_indices, high_indices, low_counts, high_counts


def _safe_average(totals, counts):
    safe_totals = tf.where(~tf.equal(counts, 0), totals, 0)
    return tf.where(~tf.equal(counts, 0), safe_totals / counts, 0)


def _flip_patches_no_diag(inp_matrix, size_x, size_y):

    tf.debugging.assert_rank(inp_matrix, rank=4, message="func requires matrix dim struct Bsize, W, H, C")
    tf.debugging.assert_equal(tf.shape(inp_matrix)[1], tf.shape(inp_matrix)[2], message="func only supports square matrices")

    patch_diag_elem_matrix = tf.cast(tf.reshape(tf.range(start=0, limit=tf.size(inp_matrix) / (size_x * size_y)),
                       shape=(tf.shape(inp_matrix)[1] / size_x, tf.shape(inp_matrix)[2] / size_y)), dtype=tf.int32)
    patch_diag_elems = tf.linalg.diag_part(patch_diag_elem_matrix)
    no_patch_diag_elems = tf.sets.difference(tf.reshape(patch_diag_elem_matrix, shape=(1, -1)), patch_diag_elems[tf.newaxis, :]).values

    yolo2 = tf.image.extract_patches(images=inp_matrix, sizes=(1, size_x, size_y, 1), strides=(1, size_x, size_y, 1),
                                         rates=[1, 1, 1, 1], padding='VALID')
    old_old_yolo2_shape = tf.shape(yolo2)

    yolo3 = tf.gather(params=yolo2, indices=[0, 2, 1, 3], axis=-1)

    yolo2 = tf.reshape(yolo2, shape=(tf.shape(yolo2)[0], -1, tf.shape(yolo2)[-1]))
    yolo3 = tf.reshape(yolo3, shape=(tf.shape(yolo3)[0], -1, tf.shape(yolo3)[-1]))

    old_yolo2_shape = tf.gather(tf.shape(yolo2), indices=[1, 0, 2], axis=-1)
    old_yolo3_shape = tf.gather(tf.shape(yolo3), indices=[1, 0, 2], axis=-1)

    yolo2 = tf.transpose(tf.gather(params=yolo2, indices=patch_diag_elems, axis=1), perm=[1, 0, 2])
    yolo3 = tf.transpose(tf.gather(params=yolo3, indices=no_patch_diag_elems, axis=1), perm=[1, 0, 2])

    yolo2 = tf.scatter_nd(indices=patch_diag_elems[:, tf.newaxis], updates=yolo2, shape=old_yolo2_shape)
    yolo3 = tf.scatter_nd(indices=no_patch_diag_elems[:, tf.newaxis], updates=yolo3, shape=old_yolo3_shape)

    yolo4 = tf.cast(tf.reshape(tf.transpose(tf.add(yolo2, yolo3), perm=[1, 0, 2]), shape=old_old_yolo2_shape), dtype=tf.float32)

    @tf.function
    def extract_patches_inverse(shape, patches, loc_size_x, loc_size_y):
        _x = tf.zeros(shape)
        _y = tf.image.extract_patches(
            _x,
            (1, loc_size_x, loc_size_y, 1),
            (1, loc_size_x, loc_size_y, 1),
            (1, 1, 1, 1),
            padding="VALID")
        grad = tf.gradients(_y, _x)[0]
        return tf.cast(tf.gradients(_y, _x, grad_ys=patches)[0] / grad, dtype=tf.int32)

    return extract_patches_inverse(tf.shape(inp_matrix), yolo4, size_x, size_y)
