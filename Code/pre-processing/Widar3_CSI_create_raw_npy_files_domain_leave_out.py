import os
import h5py
import itertools
import numpy as np
import multiprocessing
from scipy import signal
from CSIKit.reader import IWLBeamformReader


def process_item(current_root, files_one_item):
    first_item = files_one_item[0]

    item_name_split = first_item.split("-")
    item_name_split[0] = item_name_split[0].replace("user", "")
    item_name_split[-1] = item_name_split[-1].replace(".dat", "")
    item_name_split[-1] = item_name_split[-1].replace("r", "")
    item_name_split = list(map(int, item_name_split))

    if current_root.split(os.sep)[4] == "20181109":
        room_label = 1

        item_name_split[1] = 10 if item_name_split[1] == 5 else (
            11 if item_name_split[1] == 6 else item_name_split[1]
        )
    elif current_root.split(os.sep)[4] == "20181112" or current_root.split(os.sep)[4] == "20181116":
        room_label = 1

        item_name_split[1] += 12
    elif current_root.split(os.sep)[4] == "20181115":
        room_label = 1

        item_name_split[1] = 12 if item_name_split[1] == 4 else (
            10 if item_name_split[1] == 5 else (
                11 if item_name_split[1] == 6 else item_name_split[1]
            )
        )
    elif current_root.split(os.sep)[4] == "20181117" or current_root.split(os.sep)[4] == "20181118":
        room_label = 2

        item_name_split[1] = 10 if item_name_split[1] == 5 else (
            11 if item_name_split[1] == 6 else (
                12 if item_name_split[1] == 4 else item_name_split[1]
            )
        )
    elif current_root.split(os.sep)[4] == "20181121" or current_root.split(os.sep)[4] == "20181127":
        room_label = 1
        if current_root.split(os.sep)[4] == "20181127":
            room_label = 2

        item_name_split[1] = 4 if item_name_split[1] == 1 else (
            6 if item_name_split[1] == 2 else (
                9 if item_name_split[1] == 3 else (
                    5 if item_name_split[1] == 4 else (
                        8 if item_name_split[1] == 5 else (
                            7 if item_name_split[1] == 6 else item_name_split[1]
                        )
                    )
                )
            )
        )
    elif current_root.split(os.sep)[4] == "20181128":
        room_label = 2

        item_name_split[1] = 6 if item_name_split[1] == 4 else (
            9 if item_name_split[1] == 5 else (
                5 if item_name_split[1] == 6 else item_name_split[1]
            )
        )
    elif current_root.split(os.sep)[4] == "20181130" or current_root.split(os.sep)[4] == "20181204":
        room_label = 1
        if current_root.split(os.sep)[4] == "20181204":
            room_label = 2

        item_name_split[1] = 6 if item_name_split[1] == 5 else (
            9 if item_name_split[1] == 6 else (
                5 if item_name_split[1] == 7 else (
                    7 if item_name_split[1] == 9 else item_name_split[1]
                )
            )
        )
    elif current_root.split(os.sep)[4] == "20181205":
        room_label = 2
        if item_name_split[0] == 2:
            item_name_split[1] = 6 if item_name_split[1] == 1 else (
                9 if item_name_split[1] == 2 else (
                    5 if item_name_split[1] == 3 else (
                        8 if item_name_split[1] == 4 else (
                            7 if item_name_split[1] == 5 else item_name_split[1]
                        )
                    )
                )
            )
        elif item_name_split[0] == 3:
            item_name_split[1] = 4 if item_name_split[1] == 1 else (
                6 if item_name_split[1] == 2 else (
                    9 if item_name_split[1] == 3 else (
                        5 if item_name_split[1] == 4 else (
                            8 if item_name_split[1] == 5 else (
                                7 if item_name_split[1] == 6 else item_name_split[1]
                            )
                        )
                    )
                )
            )
    elif current_root.split(os.sep)[4] == "20181208":
        room_label = 2
    elif current_root.split(os.sep)[4] == "20181209":
        room_label = 2

        if item_name_split[0] == 6:
            item_name_split[1] = 6 if item_name_split[1] == 5 else (
                9 if item_name_split[1] == 6 else item_name_split[1]
            )
    elif current_root.split(os.sep)[4] == "20181211":
        room_label = 3

        item_name_split[1] = 6 if item_name_split[1] == 5 else (
            9 if item_name_split[1] == 6 else item_name_split[1]
        )
    else:
        raise Exception("An unknown data collection date subfolder was encountered")

    # Skip specific domains/gestures for correct domain factor leave out cross-validation due to distribution imbalances
    # --------------------4500 samples (6 users 5 positions 5 orientations 6 gestures 5 instances--------------------- #

    # Limit number of repetitions per domain in the dataset (reduces size for hyperparameter tuning)
    # if item_name_split[4] > 1:
    #     return
    
    if any(item_name_split[0] == x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
        return
    if any(item_name_split[1] == x for x in [5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        return
    if any(item_name_split[2] == x for x in [6, 7, 8]):
        return

    item_name_split[0] = item_name_split[0] - 11

    item_name_split[1] = 5 if item_name_split[1] == 6 else (
        6 if item_name_split[1] == 9 else item_name_split[1]
    )

    # One-hot vector of which every index denotes unique (user, room, position, orientation) pair
    # Room label has been defined in if elif else ladder
    domain_label = np.zeros((6, 1, 5, 5), dtype=np.int8)
    domain_label[
        item_name_split[0] - 1,
        room_label - 1,
        item_name_split[2] - 1,
        item_name_split[3] - 1
    ] = 1
    domain_label = domain_label.flatten()

    task_label = np.zeros(6, dtype=np.int8)
    task_label[item_name_split[1] - 1] = 1

    raw_ampl_phase_profile_stacked = np.concatenate([process_rx_sample(current_root, x) for x in files_one_item], axis=1)
    raw_ampl_phase_samples = np.transpose(a=raw_ampl_phase_profile_stacked, axes=(0, 2, 1)).astype(dtype=np.float32)

    lock.acquire()

    # Local test directory: C:\\TUe - PhD\\Contrastive-self-supervised-domain-independent-learning\\
    # Experiment-environment\\Datasets\\widar3.0-pca-labels.hdf5
    with h5py.File('/data/users/bberlo/widar3.0-domain-leave-out-dataset-benchmark-raw.hdf5', 'a') as f:
        if 'task_labels' in f and 'domain_labels' in f and 'inputs' in f:
            dset_1 = f['inputs']
            dset_2 = f['task_labels']
            dset_3 = f['domain_labels']

            dset_1.resize(dset_1.shape[0] + 1, axis=0)
            dset_2.resize(dset_2.shape[0] + 1, axis=0)
            dset_3.resize(dset_3.shape[0] + 1, axis=0)

            dset_1[-1] = raw_ampl_phase_samples
            dset_2[-1] = task_label
            dset_3[-1] = domain_label
        else:
            dset_1 = f.create_dataset("inputs", (1, 2000, 60, 12), dtype="float32", maxshape=(None, 2000, 60, 12))
            dset_2 = f.create_dataset("task_labels", (1, 6), dtype="int8", maxshape=(None, 6))
            dset_3 = f.create_dataset("domain_labels", (1, 150), dtype="int8", maxshape=(None, 150))

            dset_1[0] = raw_ampl_phase_samples
            dset_2[0] = task_label
            dset_3[0] = domain_label

    lock.release()

    # --------------------10125 samples (9 users 5 positions 5 orientations 9 gestures 5 instances-------------------- #
    """
    # Limit number of repetitions per domain in the dataset (reduces size for hyperparameter tuning)
    # if item_name_split[4] > 1:
    #    return
    
    if any(item_name_split[0] == x for x in [1, 2, 3, 4, 6, 7, 8, 9]):
        return
    if any(item_name_split[1] == x for x in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        return
    if any(item_name_split[2] == x for x in [6, 7, 8]):
        return
    if room_label != 1:
        return

    if item_name_split[0] == 5:
        item_name_split[0] = 1
    else:
        item_name_split[0] -= 8

    # One-hot vector of which every index denotes unique (user, room, position, orientation) pair
    # Room label has been defined in if elif else ladder
    domain_label = np.zeros((9, 1, 5, 5), dtype=np.int8)
    domain_label[
        item_name_split[0] - 1,
        room_label - 1,
        item_name_split[2] - 1,
        item_name_split[3] - 1
    ] = 1
    domain_label = domain_label.flatten()

    task_label = np.zeros(9, dtype=np.int8)
    task_label[item_name_split[1] - 1] = 1

    raw_ampl_phase_profile_stacked = np.concatenate([process_rx_sample(current_root, x) for x in files_one_item], axis=1)
    raw_ampl_phase_samples = np.transpose(a=raw_ampl_phase_profile_stacked, axes=(0, 2, 1)).astype(dtype=np.float32)

    lock.acquire()

    # Local test directory: C:\\TUe - PhD\\Contrastive-self-supervised-domain-independent-learning\\
    # Experiment-environment\\Datasets\\widar3.0-pca-labels.hdf5
    with h5py.File('/home/mcs001/20183777/Contrastive-self-supervised-domain-independent-learning/' +
                   'Experiment-environment/Datasets/widar3.0-domain-leave-out-dataset-more-users-gestures.hdf5', 'a') as f:
        if 'task_labels' in f and 'domain_labels' in f and 'inputs' in f:
            dset_1 = f['inputs']
            dset_2 = f['task_labels']
            dset_3 = f['domain_labels']

            dset_1.resize(dset_1.shape[0] + 1, axis=0)
            dset_2.resize(dset_2.shape[0] + 1, axis=0)
            dset_3.resize(dset_3.shape[0] + 1, axis=0)

            dset_1[-1] = raw_ampl_phase_samples
            dset_2[-1] = task_label
            dset_3[-1] = domain_label
        else:
            dset_1 = f.create_dataset("inputs", (1, 2000, 60, 12), dtype="float32", maxshape=(None, 2000, 60, 12))
            dset_2 = f.create_dataset("task_labels", (1, 9), dtype="int8", maxshape=(None, 9))
            dset_3 = f.create_dataset("domain_labels", (1, 225), dtype="int8", maxshape=(None, 225))

            dset_1[0] = raw_ampl_phase_samples
            dset_2[0] = task_label
            dset_3[0] = domain_label

    lock.release()
    """

    # -----------------12000 samples (17 users 3 rooms 5 positions 5 orientations 6 gestures 5 instances-------------- #
    """
    # Limit number of repetitions per domain in the dataset (reduces size for hyperparameter tuning)
    if item_name_split[4] > 5:
        return

    if any(item_name_split[1] == x for x in [5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        return
    if any(item_name_split[2] == x for x in [6, 7, 8]):
        return

    item_name_split[1] = 5 if item_name_split[1] == 6 else (
        6 if item_name_split[1] == 9 else item_name_split[1]
    )

    # One-hot vector of which every index denotes unique (user, room, position, orientation) pair
    # Room label has been defined in if elif else ladder
    domain_label = np.zeros((17, 3, 5, 5), dtype=np.int8)
    domain_label[
        item_name_split[0] - 1,
        room_label - 1,
        item_name_split[2] - 1,
        item_name_split[3] - 1
    ] = 1
    domain_label = domain_label.flatten()

    task_label = np.zeros(6, dtype=np.int8)
    task_label[item_name_split[1] - 1] = 1

    raw_ampl_phase_profile_stacked = np.concatenate([process_rx_sample(current_root, x) for x in files_one_item], axis=1)
    raw_ampl_phase_samples = np.transpose(a=raw_ampl_phase_profile_stacked, axes=(0, 2, 1)).astype(dtype=np.float32)

    lock.acquire()

    # Local test directory: C:\\TUe - PhD\\Contrastive-self-supervised-domain-independent-learning\\
    # Experiment-environment\\Datasets\\widar3.0-pca-labels.hdf5
    with h5py.File('/home/mcs001/20183777/Contrastive-self-supervised-domain-independent-learning/' +
                   'Experiment-environment/Datasets/widar3.0-domain-leave-out-dataset-environment.hdf5', 'a') as f:
        if 'task_labels' in f and 'domain_labels' in f and 'inputs' in f:
            dset_1 = f['inputs']
            dset_2 = f['task_labels']
            dset_3 = f['domain_labels']

            dset_1.resize(dset_1.shape[0] + 1, axis=0)
            dset_2.resize(dset_2.shape[0] + 1, axis=0)
            dset_3.resize(dset_3.shape[0] + 1, axis=0)

            dset_1[-1] = raw_ampl_phase_samples
            dset_2[-1] = task_label
            dset_3[-1] = domain_label
        else:
            dset_1 = f.create_dataset("inputs", (1, 2000, 60, 12), dtype="float32", maxshape=(None, 2000, 60, 12))
            dset_2 = f.create_dataset("task_labels", (1, 6), dtype="int8", maxshape=(None, 6))
            dset_3 = f.create_dataset("domain_labels", (1, 1275), dtype="int8", maxshape=(None, 1275))

            dset_1[0] = raw_ampl_phase_samples
            dset_2[0] = task_label
            dset_3[0] = domain_label

    lock.release()
    """


def process_rx_sample(a_root, item):
    # Reading raw CSI data
    reader_obj = IWLBeamformReader()
    data_obj = reader_obj.read_file(a_root + os.sep + item)
    csi_matrix = np.stack([x.csi_matrix for x in data_obj.frames])
    csi_matrix = np.squeeze(csi_matrix)
    csi_matrix = np.transpose(csi_matrix, [2, 0, 1])

    # Phase unwrapping
    csi_matrix_abs = np.abs(csi_matrix)
    csi_matrix_ang = np.angle(csi_matrix)
    csi_matrix_ang_unwrap = np.unwrap(csi_matrix_ang, axis=1)
    csi_matrix_ang_unwrap = np.unwrap(csi_matrix_ang_unwrap, axis=2)
    csi_matrix = csi_matrix_abs * np.exp(1j * csi_matrix_ang_unwrap)

    # First, second antenna pair selection (WiDance https://dl.acm.org/doi/pdf/10.1145/3025453.3025678)
    amplitude_mean = np.mean(np.absolute(csi_matrix), axis=1)
    amplitude_var = np.sqrt(np.var(np.absolute(csi_matrix), axis=1))
    mean_var_ratio = np.divide(amplitude_mean, amplitude_var)
    mean_var_ratio = np.mean(mean_var_ratio, axis=1)
    max_idx = np.argmax(mean_var_ratio)
    csi_matrix_ref = np.stack([csi_matrix[max_idx]] * 3)
    max_idx += 1

    # Antenna power adjustment (IndoTrack https://dl.acm.org/doi/pdf/10.1145/3130940)
    amplitude = np.absolute(csi_matrix)
    amplitude_mask = np.ma.masked_equal(amplitude, value=0.0, copy=False)
    alpha = amplitude_mask.min(axis=1)
    amplitude = amplitude - np.transpose(np.stack([alpha] * amplitude.shape[1]), [1, 0, 2])
    amplitude = np.absolute(amplitude)
    angle = np.angle(csi_matrix)
    csi_matrix = np.multiply(amplitude, np.exp(np.multiply(1j, angle)))

    beta = np.divide(np.multiply(1000, np.sum(alpha)), alpha.size)
    amplitude_2 = np.absolute(csi_matrix_ref)
    amplitude_2 = np.add(amplitude_2, beta)
    angle_2 = np.angle(csi_matrix_ref)
    csi_matrix_ref = np.multiply(amplitude_2, np.exp(np.multiply(1j, angle_2)))

    # Conjugate multiplication (DataPort DFSExtraction matlab script)
    conj_multiplication = np.multiply(csi_matrix, np.conjugate(csi_matrix_ref))
    conj_multiplication = np.transpose(conj_multiplication, [1, 0, 2])
    conj_multiplication_old_shape = conj_multiplication.shape

    conj_multiplication = np.reshape(conj_multiplication, newshape=(conj_multiplication.shape[0],
                                     conj_multiplication.shape[1] * conj_multiplication.shape[2]))
    conj_multiplication = np.concatenate(
        (conj_multiplication[:, 0:30 * (max_idx - 1)], conj_multiplication[:, 30 * max_idx:90]), axis=-1)

    # Static/high frequency component filtering (DataPort DFSExtraction matlab script)
    [lb, la] = signal.butter(6, 60 / 500, 'low')
    [hb, ha] = signal.butter(3, 2 / 500, 'high')
    conj_multiplication = signal.lfilter(lb, la, conj_multiplication, axis=0)
    conj_multiplication = signal.lfilter(hb, ha, conj_multiplication, axis=0)

    # Create raw amplitude/phase tensor
    conj_multiplication = np.reshape(conj_multiplication, newshape=(conj_multiplication.shape[0],
                                     conj_multiplication_old_shape[1] - 1, conj_multiplication_old_shape[2]))
    conj_multiplication = conj_multiplication[:2000, :, :]
    raw_ampl_phase_profile = np.zeros(shape=(2000, conj_multiplication.shape[1], conj_multiplication.shape[2] * 2), dtype=np.float64)
    raw_ampl_phase_profile[:conj_multiplication.shape[0], :conj_multiplication.shape[1], :conj_multiplication.shape[2]] = np.abs(conj_multiplication)
    raw_ampl_phase_profile[:conj_multiplication.shape[0], :conj_multiplication.shape[1], conj_multiplication.shape[2]:] = np.angle(conj_multiplication)

    return raw_ampl_phase_profile


def filter_list_item(list_item_root, list_item):
    if '.baiduyun.uploading.cfg' in list_item or '.dat' not in list_item:
        return False

    # list_item_root.split(os.sep)[5] for HPC, list_item_root.split(os.sep)[1] for local test directory
    if "2018" not in list_item_root.split(os.sep)[4]:
        return False
    elif list_item_root.split(os.sep)[4] == "20181109" and any(x in list_item for x in ['user2-6-4-4-2-', 'user3-1-3-1-8-']):
        return False
    elif list_item_root.split(os.sep)[4] == "20181118" and 'user2-3-5-3-4-' in list_item:
        return False
    elif list_item_root.split(os.sep)[4] == "20181209" and 'user6-3-1-1-5-' in list_item:
        return False
    elif list_item_root.split(os.sep)[4] == "20181211" and any(
            x in list_item for x in ['user8-1-1-1-1-', 'user8-3-3-3-5-', 'user9-1-1-1-1-']):
        return False

    return True


def init(a_lock):
    global lock
    lock = a_lock


if __name__ == "__main__":
    main_lock = multiprocessing.Lock()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init, initargs=(main_lock,)) as p:

        for root, _, files in os.walk("/backup2/pverhoev/widar3.0", topdown=True):
            if len(files) == 0:
                continue

            files.sort()
            files = list(filter(lambda x: filter_list_item(root, x), files))

            if len(files) != 0:
                files = iter(files)
                grouped_files = iter(lambda: list(itertools.islice(files, 6)), [])
                p.starmap(func=process_item, iterable=zip(itertools.repeat(root), grouped_files))
