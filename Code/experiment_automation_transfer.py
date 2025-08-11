from custom_items.utilities import domain_to_be_left_out_indices_calculation
from custom_items.data_fetching import fetch_labels_indices
from sklearn.model_selection import KFold
from subprocess import run
import numpy as np
import argparse
import pickle
import uuid

# Command prompt settings for experiment automation
parser = argparse.ArgumentParser(description='Experiment automation setup script.')
parser.add_argument('-m_n', '--model_name', help='<Required> Model name used in the experiment: domain_class, no_domain_class, multilabel_domain_class, multilabel_domain_class_2', required=True)
parser.add_argument('-t', '--type', help='<Required> Experiment type: in-domain, domain-leave-out', required=True)
parser.add_argument('-do_t', '--domain_type', help='<Required> Domain type: user, position, orientation', required=False)
parser.add_argument('-d_t', '--data_type', help='<Required> Data type used during experiment: dfs, gaf', required=True)
parser.add_argument('-d', '--dataset', help='<Required> Dataset used during experiment: widar3, signfi', required=True)
parser.add_argument('-h_a', '--half', type=int, help='Set experiment list half to be used in the experiment', required=True)
parser.add_argument('-g', '--gpu', type=int, help='When tuning, set GPU since only one cross validation fold is used', required=False)
parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=False)
args = parser.parse_args()

# Fetch domain labels
if args.data_type == 'ampphase':
    if args.dataset == 'signfi' and args.domain_type == 'user':
        dataset_path = r'Datasets/widar3.0-domain-leave-out-dataset-benchmark-raw.hdf5'
        trans_dataset_path = r'Datasets/signfi-domain-leave-out-dataset-raw-user.hdf5'
        domain_loss_weight = 0.1
    elif args.dataset == 'signfi' and args.domain_type == 'environment':
        dataset_path = r'Datasets/widar3.0-domain-leave-out-dataset-benchmark-raw.hdf5'
        trans_dataset_path = r'Datasets/signfi-domain-leave-out-dataset-raw-environment.hdf5'
        domain_loss_weight = 0.1
    elif args.dataset == 'widar3':
        dataset_path = r'Datasets/signfi-domain-leave-out-dataset-raw-user.hdf5'
        trans_dataset_path = r'Datasets/widar3.0-domain-leave-out-dataset-benchmark-raw.hdf5'
        domain_loss_weight = 0.01
    else:
        raise ValueError("Unknown dataset or domain type: {}, {}".format(args.dataset, args.domain_type))

else:
    raise ValueError("Unknown data_type. Allowed values: ampphase.")

domain_labels = fetch_labels_indices(f_path=trans_dataset_path)
domain_labels = np.argmax(domain_labels, axis=1) + 1

pretrain_domain_labels = fetch_labels_indices(f_path=dataset_path)
pretrain_indices = np.arange(start=0, stop=pretrain_domain_labels.shape[0])

# Currently not supported: domain_aware_batch, domain_aware_filter_denom, domain_aware_batch_alt_flow, domain_aware_filter_denom_alt_flow
if args.model_name == 'domain_class':
    f_name = 'experiment_domain_classification.py'
elif args.model_name == 'no_domain_class':
    f_name = 'experiment_no_domain_classification.py'
elif args.model_name == 'multilabel_domain_class':
    f_name = 'experiment_multilabel_domain_classification.py'
elif args.model_name == 'multilabel_domain_class_2':
    f_name = 'experiment_multilabel_domain_classification_2.py'
elif args.model_name == 'supervised_no_domain_class':
    f_name = 'experiment_supervised_no_domain_classification_transfer.py'
else:
    raise ValueError("Unknown model_name. Allowed values: domain_class, no_domain_class, multilabel_domain_class, multilabel_domain_class_2.")

if args.type == 'domain-leave-out':
    CUR_DOMAIN_FACTOR_NAME = args.domain_type

    if args.dataset == 'signfi':
        if args.domain_type == 'user':
            domain_label_struct = (5, 1, 1, 1)
            CUR_DOMAIN_FACTOR = 0
            CUR_DOMAIN_FACTOR_TOTAL_NR = 5
        elif args.domain_type == 'environment':
            domain_label_struct = (2, 1, 1, 1)
            CUR_DOMAIN_FACTOR = 0
            CUR_DOMAIN_FACTOR_TOTAL_NR = 2
        else:
            raise ValueError("Unknown domain type. Allowed values: user, environment")

    elif args.dataset == 'widar3':
        domain_label_struct = (6, 1, 5, 5)

        if args.domain_type == 'user':
            CUR_DOMAIN_FACTOR = 0
            CUR_DOMAIN_FACTOR_TOTAL_NR = 6
        elif args.domain_type == 'environment':
            CUR_DOMAIN_FACTOR = 1
            CUR_DOMAIN_FACTOR_TOTAL_NR = 1
        else:
            raise ValueError("Unknown domain type. Allowed values: user, environment.")

    else:
        raise ValueError("Unknown dataset type: {}.".format(args.dataset))

    if CUR_DOMAIN_FACTOR_TOTAL_NR > 1:

        # First func. variable needs to match loc. of CUR_TOTAL_DOMAIN_FACTOR_NR in quadruple
        test_types_list = [list(map(lambda y: y + 1, domain_to_be_left_out_indices_calculation(CUR_DOMAIN_FACTOR, x,
        domain_label_struct))) for x in range(CUR_DOMAIN_FACTOR_TOTAL_NR)]

    else:

        kfold_obj = KFold(n_splits=5, shuffle=True, random_state=42)
        domain_types = list(range(1, np.prod(domain_label_struct).item() + 1, 1))
        test_types_indices_list = [x[1].tolist() for x in kfold_obj.split(domain_types)]
        test_types_list = [[domain_types[x] for x in z] for z in test_types_indices_list]
        test_types_list = test_types_list[:1]

else:
    raise ValueError("Unknown experiment type was given as argument.")

test_types_list_half = len(test_types_list) // 2
if args.half == 0:
    split_nrs = range(test_types_list_half)
    test_types_list = test_types_list[:test_types_list_half]
    GPU_DEVICE = 0
else:
    split_nrs = range(test_types_list_half, len(test_types_list))
    test_types_list = test_types_list[test_types_list_half:]
    GPU_DEVICE = 1

if args.crossval_split:

    # Limit experiment scope to selected crossval. split
    if args.crossval_split not in split_nrs:
        raise ValueError("Selected split does not match GPU argument. Please edit args.gpu or args.crossval_split.")
    else:
        test_types_list = [test_types_list[split_nrs.index(args.crossval_split)]]
        split_nrs = [args.crossval_split]

for split_nr, test_types in zip(split_nrs, test_types_list):
    train_types = list(set(range(1, np.prod(domain_label_struct).item() + 1, 1)) - set(test_types))
    train_indices, test_indices = \
        np.where(np.isin(domain_labels, test_elements=np.asarray(train_types)))[0], \
        np.where(np.isin(domain_labels, test_elements=np.asarray(test_types)))[0]

    if args.model_name == "domain_class" or args.model_name == "multilabel_domain_class" or args.model_name == "multilabel_domain_class_2":
        file_path = 'tmp/' + uuid.uuid4().hex + '.pickle'
        with open(file_path, 'wb') as handle:
            pickle.dump(
                obj={'train_indices': train_indices, 'pretrain_indices': pretrain_indices,
                     'test_indices': test_indices, 'train_types': train_types},
                file=handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Streams/{}_{}_{}_stdout.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stdoutFile, \
             open('Streams/{}_{}_{}_stderr.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stderrFile:

            run([
                'python', f_name,
                '-e_s', '100',
                '-b_s', '32',
                '-s', '42',
                '-d_f_n', CUR_DOMAIN_FACTOR_NAME,
                '-d_t', args.data_type,
                '-do_t', args.domain_type,
                '-d', args.dataset,
                '-cv_s', str(split_nr),
                '-g', str(GPU_DEVICE),
                '-t', '0.1',
                '-l_r', '0.0001',
                '-d_l_w', str(domain_loss_weight),
                '-m_n', args.model_name,
                '-f_p', file_path,
                '--transfer'
            ], stdout=stdoutFile, stderr=stderrFile)

    elif args.model_name == "no_domain_class":
        file_path = 'tmp/' + uuid.uuid4().hex + '.pickle'
        with open(file_path, 'wb') as handle:
            pickle.dump(
                obj={'train_indices': train_indices, 'pretrain_indices': pretrain_indices,
                     'test_indices': test_indices, 'train_types': train_types},
                file=handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Streams/{}_{}_{}_stdout.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stdoutFile, \
             open('Streams/{}_{}_{}_stderr.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stderrFile:

            run([
                'python', f_name,
                '-e_s', '100',
                '-b_s', '32',
                '-s', '42',
                '-d_f_n', CUR_DOMAIN_FACTOR_NAME,
                '-d_t', args.data_type,
                '-do_t', args.domain_type,
                '-d', args.dataset,
                '-cv_s', str(split_nr),
                '-g', str(GPU_DEVICE),
                '-t', '0.1',
                '-l_r', '0.0001',
                '-m_n', args.model_name,
                '-f_p', file_path,
                '--transfer'
            ], stdout=stdoutFile, stderr=stderrFile)

    elif args.model_name == "supervised_no_domain_class":
        file_path = 'tmp/' + uuid.uuid4().hex + '.pickle'
        with open(file_path, 'wb') as handle:
            pickle.dump(
                obj={'train_indices': train_indices, 'pretrain_indices': pretrain_indices,
                     'test_indices': test_indices, 'train_types': train_types},
                file=handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Streams/{}_{}_{}_stdout.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stdoutFile, \
             open('Streams/{}_{}_{}_stderr.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stderrFile:

            run([
                'python', f_name,
                '-e_s', '100',
                '-b_s', '16',
                '-s', '42',
                '-d_f_n', CUR_DOMAIN_FACTOR_NAME,
                '-d_t', args.data_type,
                '-do_t', args.domain_type,
                '-d', args.dataset,
                '-cv_s', str(split_nr),
                '-g', str(GPU_DEVICE),
                '-l_r', '0.0001',
                '-m_n', args.model_name,
                '-f_p', file_path
            ], stdout=stdoutFile, stderr=stderrFile)

    else:
        raise Exception("An unknown model experiment script was encountered")
