import h5py
import argparse
import itertools
import numpy as np
import multiprocessing
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import firwin, filtfilt


def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def process_item(samples, task_label, domain_label, dset_path):

    freq_time_profile = process_rx_sample(samples).astype(np.float32)

    lock.acquire()

    with h5py.File(dset_path, 'a') as f:
        if 'task_labels' in f and 'domain_labels' in f and 'inputs' in f:
            dset_1 = f['inputs']
            dset_2 = f['task_labels']
            dset_3 = f['domain_labels']

            dset_1.resize(dset_1.shape[0] + 1, axis=0)
            dset_2.resize(dset_2.shape[0] + 1, axis=0)
            dset_3.resize(dset_3.shape[0] + 1, axis=0)

            dset_1[-1] = freq_time_profile
            dset_2[-1] = task_label
            dset_3[-1] = domain_label
        else:
            dset_1 = f.create_dataset("inputs", (1, 200, 60, 3), dtype="float32", maxshape=(None, 200, 60, 3))
            dset_2 = f.create_dataset("task_labels", (1, task_label.shape[-1]), dtype="int16", maxshape=(None, task_label.shape[-1]))
            dset_3 = f.create_dataset("domain_labels", (1, domain_label.shape[-1]), dtype="int16", maxshape=(None, domain_label.shape[-1]))

            dset_1[0] = freq_time_profile
            dset_2[0] = task_label
            dset_3[0] = domain_label

    lock.release()


def process_rx_sample(item):
    raw_frame_list = []

    for matrix_per_tx_antenna in np.transpose(item, axes=[2, 0, 1]):

        matrix_per_tx_antenna_abs = np.abs(matrix_per_tx_antenna)
        matrix_per_tx_antenna_angle = np.angle(matrix_per_tx_antenna)
        raw_frame_list.append(np.concatenate((matrix_per_tx_antenna_abs, matrix_per_tx_antenna_angle), axis=-1))

    return np.stack(raw_frame_list, axis=-1)


def func(x, a, b, c):
    """Phase offsets function
    x[0]: transmit antenna index
    x[1]: subcarrier index
    """
    return (a*x[0] + b*x[1] + c).astype(np.float64)


def init(a_lock):
    global lock
    lock = a_lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SignFi CSI pre-processing script.')
    parser.add_argument('-t', '--type', help='<Required> Input dataset type', required=True)
    args = parser.parse_args()

    if args.type == 'env_home':
        signfi_dataset = loadmat('raw/dataset_home_276.mat', appendmat=False)
        signfi_samples = np.transpose(signfi_dataset["csid_home"], axes=[3, 0, 1, 2])
        signfi_onehot_labels = to_categorical(signfi_dataset["label_home"].flatten() - 1, num_classes=276, dtype='int16')
        signfi_onehot_domain_labels = to_categorical(np.full(shape=(signfi_onehot_labels.shape[0],), fill_value=0), num_classes=2, dtype='int16')
        d_path = 'processed/signfi_raw_environment.hdf5'

    elif args.type == 'env_lab':
        signfi_dataset = loadmat('raw/dataset_lab_276_dl.mat', appendmat=False)
        signfi_samples = np.transpose(signfi_dataset["csid_lab"], axes=[3, 0, 1, 2])
        signfi_onehot_labels = to_categorical(signfi_dataset["label_lab"].flatten() - 1, num_classes=276, dtype='int16')
        signfi_onehot_domain_labels = to_categorical(np.full(shape=(signfi_onehot_labels.shape[0],), fill_value=1), num_classes=2, dtype='int16')
        d_path = 'processed/signfi_raw_environment.hdf5'

    elif args.type == 'user_lab':

        signfi_dataset = loadmat('raw/dataset_lab_150.mat', appendmat=False)
        signfi_samples = np.concatenate([np.transpose(signfi_dataset["csi{}".format(elem)], axes=[3, 0, 1, 2]) for elem in [1, 2, 3, 4, 5]], axis=0)

        signfi_number_labels = signfi_dataset["label"]
        signfi_number_labels[signfi_number_labels > 125] = signfi_number_labels[signfi_number_labels > 125] - 128
        signfi_onehot_labels = to_categorical(signfi_number_labels.flatten() - 1, num_classes=150, dtype='int16')

        signfi_onehot_domain_labels = np.concatenate([to_categorical(np.full(shape=(signfi_onehot_labels.shape[0] // 5,), fill_value=elem), num_classes=5, dtype='int16') for elem in [0, 1, 2, 3, 4]], axis=0)
        d_path = 'processed/signfi_raw_user.hdf5'

    else:
        raise TypeError('Encountered unknown dataset type. Possible values: env_home, env_lab, and user_lab')

    signfi_samples_shape = signfi_samples.shape

    # x (tx and subcarrier index) for curve fitting
    idx_tx_subc = np.zeros(shape=(2, 3, 30, signfi_samples_shape[0]))
    init_idx_tx_subc_shape = idx_tx_subc.shape

    for tx in range(3):
        for k in range(30):
            for n in range(signfi_samples_shape[0]):
                idx_tx_subc[0, tx, k, n] = (tx + 2) / 3 - 2  # tx index, reordered
                idx_tx_subc[1, tx, k, n] = -58 + 4 * k  # sub-carrier index

    idx_tx_subc = np.reshape(idx_tx_subc, (2, -1))

    signfi_samples_abs = np.abs(signfi_samples)
    signfi_samples_ang = np.angle(signfi_samples)
    signfi_samples_ang_unwrap = np.unwrap(signfi_samples_ang, axis=1)
    signfi_samples_ang_unwrap = np.unwrap(signfi_samples_ang_unwrap, axis=2)

    popt, _ = curve_fit(func, idx_tx_subc.astype(np.float64), np.transpose(signfi_samples_ang_unwrap[:, 0, :, :], axes=[1, 2, 0]).flatten().astype(np.float64))
    phase_correction_tensor = np.broadcast_to(np.reshape(func(np.reshape(idx_tx_subc, newshape=init_idx_tx_subc_shape)[:, :, :, 0], *popt), newshape=(signfi_samples_shape[2:4]))[np.newaxis, np.newaxis, :, :], shape=signfi_samples_shape)
    signfi_samples_ang_unwrap -= phase_correction_tensor

    fir_filt_coeff = firwin(numtaps=4, cutoff=[2, 80], window='hamming', pass_zero='bandpass', fs=200)
    signfi_samples_ang_unwrap_filt = filtfilt(b=fir_filt_coeff, a=1, x=signfi_samples_ang_unwrap, axis=1)

    new_signfi_samples = signfi_samples_abs * np.exp(1j * signfi_samples_ang_unwrap_filt)

    main_lock = multiprocessing.Lock()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init, initargs=(main_lock,)) as p:
        p.starmap(func=process_item, iterable=zip(iter(new_signfi_samples), iter(signfi_onehot_labels), iter(signfi_onehot_domain_labels), itertools.repeat(d_path)))
