import sklearn.model_selection as sk
from custom_items.augmenters import augment
import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
import h5py


def fetch_labels_indices(f_path, indices=None, domain_types=None, fine_tune=False, seed=None, val_set_sampling=False):
    if indices is None:
        with h5py.File(f_path, 'r') as f:
            dset_1 = f['domain_labels']
            widar_domain_labels = dset_1[:]

        return widar_domain_labels

    elif indices is not None and fine_tune is True:
        with h5py.File(f_path, 'r') as f:
            dset_1 = f['task_labels']
            all_widar_task_labels = dset_1[:]

        widar_task_labels = all_widar_task_labels[indices]

        if val_set_sampling is True:
            split_nr = 2
        else:
            split_nr = 4

        k_fold_object = sk.StratifiedKFold(n_splits=split_nr, shuffle=True, random_state=seed)
        widar_sparse_task_labels = np.argmax(widar_task_labels, axis=1)

        train_indices, val_indices = next(k_fold_object.split(widar_task_labels, widar_sparse_task_labels))
        return train_indices, val_indices

    else:
        with h5py.File(f_path, 'r') as f:
            dset_1 = f['domain_labels']
            all_widar_domain_labels = dset_1[:]

        widar_domain_labels = all_widar_domain_labels[indices]

        k_fold_object = sk.KFold(n_splits=6, shuffle=True, random_state=seed)
        widar_sparse_domain_labels = np.argmax(widar_domain_labels, axis=1) + 1
        domain_types = np.asarray(domain_types)

        train_type_indices, val_type_indices = next(k_fold_object.split(X=np.expand_dims(a=domain_types, axis=1)))
        train_types, val_types = domain_types[train_type_indices], domain_types[val_type_indices]

        train_indices, val_indices = \
            np.where(np.isin(widar_sparse_domain_labels, test_elements=train_types))[0], \
            np.where(np.isin(widar_sparse_domain_labels, test_elements=val_types))[0]

        return train_indices, val_indices


def dataset_constructor(instances, f_path, subset_type, batch_size, seed, data_format, domain_type=None, end_to_end=False):
    if data_format == 'ampphase':
        spec = {'/inputs': tf.TensorSpec(shape=(2000, 60, 12), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(6,), dtype=tf.int8),
                '/domain_labels': tf.TensorSpec(shape=(150,), dtype=tf.int8)}
    else:
        raise ValueError("Unknown data_format. Allowed values: ampphase")

    widar_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    widar_inputs = widar_hdf5('/inputs')
    widar_task_labels = widar_hdf5('/task_labels')
    widar_domain_labels = widar_hdf5('/domain_labels')

    def get_sample(instance, label_type):

        def pad_power_2(x):
            if data_format == 'ampphase':
                return tf.image.pad_to_bounding_box(x, 24, 2, 2048, 64)
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        if label_type == 'domain_label':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance], widar_domain_labels[instance]

            if data_format == 'ampphase':
                return pad_power_2(widar_inputs[instance]), widar_domain_labels[instance]
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        elif label_type == 'task_label':
            if data_format == 'ampphase':
                if subset_type == 'pre-train' and end_to_end:
                    return widar_inputs[instance], widar_task_labels[instance]
                else:
                    return pad_power_2(widar_inputs[instance]), widar_task_labels[instance]
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        elif label_type == 'none':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance]

            if data_format == 'ampphase':
                return pad_power_2(widar_inputs[instance])
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        else:
            raise ValueError("Unknown label type definition.")

    def group_reduce_func(dataset, window_size, sub_type):

        def create_view_comb_batch(batch_inputs, batch_labels):

            reduced_input_batch_v1 = batch_inputs[:, :, :, :6]
            reduced_input_batch_v2 = batch_inputs[:, :, :, 6:]

            if not end_to_end:
                return (reduced_input_batch_v1, reduced_input_batch_v2), batch_labels
            else:
                return batch_inputs, batch_labels

        dataset = dataset.batch(window_size).map(create_view_comb_batch).prefetch(20)
        return dataset

    dset = tf.data.Dataset.from_tensor_slices(instances)

    if end_to_end:
        label_to_take = 'task_label'
    else:
        label_to_take = 'domain_label'

    if subset_type == 'pre-train':
        individual_sets = \
            [tf.data.Dataset.from_tensor_slices(instances)
                 .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat() for _ in range(3)]

        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, label_to_take))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        combined_set = tf.data.Dataset.zip(tuple(individual_sets))\
            .map(lambda x1y, x2, x3: (augment([x1y[0], x2, x3], data_format=data_format, dataset='widar3'), x1y[1]))\
            .apply(transformation_func=lambda dataset: group_reduce_func(dataset, batch_size, subset_type))

        return combined_set

    elif subset_type == 'pre-train-val':
        dset = dset.repeat().map(lambda x: get_sample(x, label_to_take))\
               .apply(transformation_func=lambda dataset: group_reduce_func(dataset, batch_size, subset_type))

    elif subset_type == 'fine-tune':
        dset = dset.shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat().map(lambda x: get_sample(x, 'task_label'))\
               .apply(transformation_func=lambda dataset: group_reduce_func(dataset, batch_size, subset_type))

    elif subset_type == 'fine-tune-val' or subset_type == 'test':
        dset = dset.repeat().map(lambda x: get_sample(x, 'task_label'))\
               .apply(transformation_func=lambda dataset: group_reduce_func(dataset, batch_size, subset_type))

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: pre-train(-val), fine-tune(-val), test.")

    return dset


def dom_aware_dataset_constructor(instances, f_path, subset_type, batch_size, seed, data_format, domain_type=None):
    if data_format == 'ampphase':
        spec = {'/inputs': tf.TensorSpec(shape=(2000, 60, 12), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(6,), dtype=tf.int8),
                '/domain_labels': tf.TensorSpec(shape=(150,), dtype=tf.int8)}
    else:
        raise ValueError("Unknown data_format. Allowed values: ampphase")

    widar_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    widar_inputs = widar_hdf5('/inputs')
    widar_task_labels = widar_hdf5('/task_labels')
    widar_domain_labels = widar_hdf5('/domain_labels')

    def get_sample(instance, label_type):

        def pad_power_2(x):
            if data_format == 'ampphase':
                return tf.image.pad_to_bounding_box(x, 24, 2, 2048, 64)
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        if label_type == 'domain_task_label':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance], widar_domain_labels[instance], widar_task_labels[instance]

            if data_format == 'ampphase':
                return pad_power_2(widar_inputs[instance]), widar_domain_labels[instance], widar_task_labels[instance]
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        elif label_type == 'none':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance]

            if data_format == 'ampphase':
                return pad_power_2(widar_inputs[instance])
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        else:
            raise ValueError("Unknown label type definition.")

    def group_reduce_func(dataset, window_size, sub_type):

        def create_view_comb(a_input, batch_label, batch_task_label):

            reduced_input__v1 = a_input[:, :, :6]
            reduced_input__v2 = a_input[:, :, 6:]

            return tf.stack([reduced_input__v1, reduced_input__v2], axis=0), \
                   tf.stack([batch_label, batch_label], axis=0), \
                   tf.stack([batch_task_label, batch_task_label], axis=0)

        def create_view_comb_2(view_inputs, view_batch_labels, view_batch_task_labels):

            return (view_inputs[0], view_inputs[1]), (view_batch_labels[0], view_batch_labels[1])

        def create_view_comb_3(batched_inputs, batched_labels):

            return (tf.stack([batched_inputs[0][0], batched_inputs[1][0]], axis=0),
                    tf.stack([batched_inputs[0][1], batched_inputs[1][1]], axis=0)),\
                   (tf.stack([batched_labels[0][0], batched_labels[1][0]], axis=0),
                    tf.stack([batched_labels[0][1], batched_labels[1][1]], axis=0))

        dataset = dataset.map(create_view_comb).unbatch()\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s, y: tf.argmax(y),
                reduce_func=lambda _, dataset: dataset.batch(2).map(create_view_comb_2),
                window_size=2
            ))\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s: tf.argmax(s[0]) + 200 * tf.argmax(s[1]),
                reduce_func=lambda _, dataset: dataset.batch(batch_size),
                window_size=batch_size)).shuffle(buffer_size=24)\
            .batch(2).map(create_view_comb_3).unbatch().shuffle(buffer_size=24)\
            .filter(lambda _, s: tf.math.reduce_all(tf.not_equal(tf.argmax(s[0], axis=-1), tf.argmax(s[1], axis=-1))))\
            .prefetch(20)

        return dataset

    dset = tf.data.Dataset.from_tensor_slices(instances)

    label_to_take = 'domain_task_label'

    if subset_type == 'pre-train':
        individual_sets = \
            [tf.data.Dataset.from_tensor_slices(instances)
                 .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat() for _ in range(3)]

        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, label_to_take))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        combined_set = tf.data.Dataset.zip(tuple(individual_sets))\
            .map(lambda x1y, x2, x3: (augment([x1y[0], x2, x3], data_format=data_format, dataset='widar3'), x1y[1], x1y[2]))\
            .apply(transformation_func=lambda dataset: group_reduce_func(dataset, batch_size, subset_type))

        return combined_set

    elif subset_type == 'pre-train-val':
        dset = dset.repeat().map(lambda x: get_sample(x, label_to_take))\
               .apply(transformation_func=lambda dataset: group_reduce_func(dataset, batch_size, subset_type))

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: pre-train(-val), fine-tune(-val), test.")

    return dset


def dataset_constructor_signfi(instances, f_path, subset_type, batch_size, seed, data_format, domain_type, end_to_end=False):
    if data_format == 'ampphase' and domain_type == 'user':
        spec = {'/inputs': tf.TensorSpec(shape=(200, 60, 3), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(150,), dtype=tf.int16),
                '/domain_labels': tf.TensorSpec(shape=(5,), dtype=tf.int16)}
    elif data_format == 'ampphase' and domain_type == 'environment':
        spec = {'/inputs': tf.TensorSpec(shape=(200, 60, 3), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(276,), dtype=tf.int16),
                '/domain_labels': tf.TensorSpec(shape=(2,), dtype=tf.int16)}
    else:
        raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

    widar_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    widar_inputs = widar_hdf5('/inputs')
    widar_task_labels = widar_hdf5('/task_labels')
    widar_domain_labels = widar_hdf5('/domain_labels')

    def get_sample(instance, label_type):

        def pad_power_2(x):
            if data_format == 'ampphase':
                return tf.image.pad_to_bounding_box(x, 12, 2, 224, 64)
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        if label_type == 'domain_task_label':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance], widar_domain_labels[instance], widar_task_labels[instance]

            return pad_power_2(widar_inputs[instance]), widar_domain_labels[instance], widar_task_labels[instance]

        elif label_type == 'none':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance]

            return pad_power_2(widar_inputs[instance])

        else:
            raise ValueError("Unknown label type definition.")

    def group_reduce_func(dataset, window_size, sub_type):

        def create_view_comb_batch(batch_inputs, batch_labels, batch_task_labels):

            reduced_input_batch_v1 = batch_inputs[0, :, :, :]
            reduced_input_batch_v2 = batch_inputs[1, :, :, :]

            if 'pre-train' in sub_type:
                return (reduced_input_batch_v1, reduced_input_batch_v2), batch_labels[0]
            else:
                raise ValueError("Grouping samples with similar task, domain labels is only allowed during pre-training")

        dataset = dataset.batch(window_size).map(create_view_comb_batch)
        return dataset

    dset = tf.data.Dataset.from_tensor_slices(instances)

    if subset_type == 'pre-train' and not end_to_end:
        individual_sets = \
            [tf.data.Dataset.from_tensor_slices(instances)
                .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat() for _ in range(3)]

        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, 'domain_task_label'))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        combined_set = tf.data.Dataset.zip(tuple(individual_sets))

        combined_set = combined_set\
            .map(lambda x1y, x2, x3: (augment([x1y[0], x2, x3], data_format=data_format, dataset='signfi'), x1y[1], x1y[2]))

        combined_set = combined_set\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s, y: tf.argmax(y) + 276 * tf.argmax(s),
                reduce_func=lambda _, dataset: group_reduce_func(dataset, 2, subset_type),
                window_size=2))\
            .batch(batch_size)\
            .prefetch(20)

        return combined_set

    elif subset_type == 'pre-train' and end_to_end:
        individual_sets = \
            [tf.data.Dataset.from_tensor_slices(instances)
                .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat() for _ in range(3)]

        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, 'domain_task_label'))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        combined_set = tf.data.Dataset.zip(tuple(individual_sets))

        combined_set = combined_set\
            .map(lambda x1y, x2, x3: (augment([x1y[0], x2, x3], data_format=data_format, dataset='signfi'), x1y[2])) \
            .batch(batch_size) \
            .prefetch(20)

        return combined_set

    elif subset_type == 'pre-train-val' and not end_to_end:
        dset = dset.repeat().map(lambda x: get_sample(x, 'domain_task_label'))
        dset = dset.apply(tf.data.experimental.group_by_window(
                    key_func=lambda _, s, y: tf.argmax(y) + 276 * tf.argmax(s),
                    reduce_func=lambda _, dataset: group_reduce_func(dataset, 2, subset_type),
                    window_size=2
               )).batch(batch_size).prefetch(20)

    elif subset_type == 'pre-train-val' and end_to_end:
        dset = dset.repeat().map(lambda x: list(get_sample(x, 'domain_task_label')[i] for i in [0, 2]))
        dset = dset.batch(batch_size).prefetch(20)

    elif subset_type == 'fine-tune':
        dset = dset.shuffle(buffer_size=len(instances), reshuffle_each_iteration=True)
        dset = dset.repeat().map(lambda x: list(get_sample(x, 'domain_task_label')[i] for i in [0, 2]))

        dset = dset.batch(batch_size).prefetch(20)

    elif subset_type == 'fine-tune-val' or subset_type == 'test':
        dset = dset.repeat().map(lambda x: list(get_sample(x, 'domain_task_label')[i] for i in [0, 2]))
        dset = dset.batch(batch_size).prefetch(20)

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: pre-train(-val), fine-tune(-val), test.")

    return dset


def dom_aware_dataset_constructor_signfi(instances, f_path, subset_type, batch_size, seed, data_format, domain_type, end_to_end=False):

    if end_to_end:
        raise ValueError("Domain awaire dataset constructor can only be used during pre-training.")

    if data_format == 'ampphase' and domain_type == 'user':
        spec = {'/inputs': tf.TensorSpec(shape=(200, 60, 3), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(150,), dtype=tf.int16),
                '/domain_labels': tf.TensorSpec(shape=(5,), dtype=tf.int16)}
    elif data_format == 'ampphase' and domain_type == 'environment':
        spec = {'/inputs': tf.TensorSpec(shape=(200, 60, 3), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(276,), dtype=tf.int16),
                '/domain_labels': tf.TensorSpec(shape=(2,), dtype=tf.int16)}
    else:
        raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

    widar_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    widar_inputs = widar_hdf5('/inputs')
    widar_task_labels = widar_hdf5('/task_labels')
    widar_domain_labels = widar_hdf5('/domain_labels')

    def get_sample(instance, label_type):

        def pad_power_2(x):
            if data_format == 'ampphase':
                return tf.image.pad_to_bounding_box(x, 12, 2, 224, 64)
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        if label_type == 'domain_task_label':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance], widar_domain_labels[instance], widar_task_labels[instance]

            return pad_power_2(widar_inputs[instance]), widar_domain_labels[instance], widar_task_labels[instance]

        elif label_type == 'none':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance]

            return pad_power_2(widar_inputs[instance])

        else:
            raise ValueError("Unknown label type definition.")

    def group_reduce_func(dataset, window_size, sub_type):

        def create_view_comb_batch(batch_inputs, batch_labels, batch_task_labels):

            reduced_input_batch_v1 = batch_inputs[0, :, :, :]
            reduced_input_batch_v2 = batch_inputs[1, :, :, :]

            if 'pre-train' in sub_type:
                return (reduced_input_batch_v1, reduced_input_batch_v2), (batch_labels[0], batch_labels[1])
            else:
                raise ValueError("Grouping samples with similar task, domain labels is only allowed during pre-training")

        dataset = dataset.batch(window_size).map(create_view_comb_batch)
        return dataset

    dset = tf.data.Dataset.from_tensor_slices(instances)

    if subset_type == 'pre-train':
        individual_sets = \
            [tf.data.Dataset.from_tensor_slices(instances)
                .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat() for _ in range(3)]

        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, 'domain_task_label'))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        combined_set = tf.data.Dataset.zip(tuple(individual_sets))

        combined_set = combined_set\
            .map(lambda x1y, x2, x3: (augment([x1y[0], x2, x3], data_format=data_format, dataset='signfi'), x1y[1], x1y[2]))

        combined_set = combined_set\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s, y: tf.argmax(y),
                reduce_func=lambda _, dataset: group_reduce_func(dataset, 2, subset_type),
                window_size=2))\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s: tf.argmax(s[0]) + 5 * tf.argmax(s[1]),
                reduce_func=lambda _, dataset: dataset.batch(batch_size),
                window_size=batch_size))\
            .filter(lambda _, s: tf.math.reduce_all(tf.not_equal(tf.argmax(s[0], axis=-1), tf.argmax(s[1], axis=-1))))\
            .prefetch(20)

        return combined_set

    elif subset_type == 'pre-train-val':
        dset = dset.repeat().map(lambda x: get_sample(x, 'domain_task_label'))
        dset = dset\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s, y: tf.argmax(y),
                reduce_func=lambda _, dataset: group_reduce_func(dataset, 2, subset_type),
                window_size=2))\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s: tf.argmax(s[0]) + 5 * tf.argmax(s[1]),
                reduce_func=lambda _, dataset: dataset.batch(batch_size),
                window_size=batch_size))\
            .filter(lambda _, s: tf.math.reduce_all(tf.not_equal(tf.argmax(s[0], axis=-1), tf.argmax(s[1], axis=-1))))\
            .prefetch(20)

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: pre-train(-val).")

    return dset


def dom_aware_dataset_constructor_signfi_2(instances, f_path, subset_type, batch_size, seed, data_format, domain_type, end_to_end=False):

    if end_to_end:
        raise ValueError("Domain awaire dataset constructor can only be used during pre-training.")

    if data_format == 'ampphase' and domain_type == 'user':
        spec = {'/inputs': tf.TensorSpec(shape=(200, 60, 3), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(150,), dtype=tf.int16),
                '/domain_labels': tf.TensorSpec(shape=(5,), dtype=tf.int16)}
    elif data_format == 'ampphase' and domain_type == 'environment':
        spec = {'/inputs': tf.TensorSpec(shape=(200, 60, 3), dtype=tf.float32),
                '/task_labels': tf.TensorSpec(shape=(276,), dtype=tf.int16),
                '/domain_labels': tf.TensorSpec(shape=(2,), dtype=tf.int16)}
    else:
        raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

    widar_hdf5 = tfio.IOTensor.from_hdf5(filename=f_path, spec=spec)
    widar_inputs = widar_hdf5('/inputs')
    widar_task_labels = widar_hdf5('/task_labels')
    widar_domain_labels = widar_hdf5('/domain_labels')

    def get_sample(instance, label_type):

        def pad_power_2(x):
            if data_format == 'ampphase':
                return tf.image.pad_to_bounding_box(x, 12, 2, 224, 64)
            else:
                raise ValueError("Unknown data_format. Allowed values: dfs, gaf.")

        if label_type == 'domain_task_label':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance], widar_domain_labels[instance], widar_task_labels[instance]

            return pad_power_2(widar_inputs[instance]), widar_domain_labels[instance], widar_task_labels[instance]

        elif label_type == 'none':
            if subset_type == 'pre-train' and data_format == 'ampphase':
                return widar_inputs[instance]

            return pad_power_2(widar_inputs[instance])

        else:
            raise ValueError("Unknown label type definition.")

    def group_reduce_func(dataset, window_size, sub_type):

        def create_view_comb_batch(batch_inputs, batch_labels, batch_task_labels):

            reduced_input_batch_v1 = batch_inputs[0, :, :, :]
            reduced_input_batch_v2 = batch_inputs[1, :, :, :]

            if 'pre-train' in sub_type:
                return (reduced_input_batch_v1, reduced_input_batch_v2), (batch_labels[0], batch_labels[1])
            else:
                raise ValueError("Grouping samples with similar task, domain labels is only allowed during pre-training")

        dataset = dataset.batch(window_size).map(create_view_comb_batch)
        return dataset

    dset = tf.data.Dataset.from_tensor_slices(instances)

    if subset_type == 'pre-train':
        individual_sets = \
            [tf.data.Dataset.from_tensor_slices(instances)
                .shuffle(buffer_size=len(instances), reshuffle_each_iteration=True).repeat() for _ in range(3)]

        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, 'domain_task_label'))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        combined_set = tf.data.Dataset.zip(tuple(individual_sets))

        combined_set = combined_set\
            .map(lambda x1y, x2, x3: (augment([x1y[0], x2, x3], data_format=data_format, dataset='signfi'), x1y[1], x1y[2]))

        combined_set = combined_set\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s, y: tf.argmax(y),
                reduce_func=lambda _, dataset: group_reduce_func(dataset, 2, subset_type),
                window_size=2))\
            .batch(batch_size)\
            .prefetch(20)

        return combined_set

    elif subset_type == 'pre-train-val':
        dset = dset.repeat().map(lambda x: get_sample(x, 'domain_task_label'))
        dset = dset\
            .apply(tf.data.experimental.group_by_window(
                key_func=lambda _, s, y: tf.argmax(y),
                reduce_func=lambda _, dataset: group_reduce_func(dataset, 2, subset_type),
                window_size=2))\
            .batch(batch_size)\
            .prefetch(20)

    else:
        raise ValueError("Unknown subset_type encountered. Allowed values: pre-train(-val).")

    return dset
