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
args = parser.parse_args()

# Note: no_domain_classification_weight_tuning.py not supported out of the box (has to be added).

# Fetch domain labels
dataset_path = r'Datasets/signfi-domain-leave-out-dataset-raw-user.hdf5'
f_name = 'supervised_no_domain_classification_tuning_scratch.py'

domain_labels = fetch_labels_indices(f_path=dataset_path)
domain_labels = np.argmax(domain_labels, axis=1) + 1

domain_label_struct = (5, 1, 1, 1)
CUR_DOMAIN_FACTOR_NAME = 'user'

kfold_obj = KFold(n_splits=5, shuffle=True, random_state=42)
domain_types = list(range(1, np.prod(domain_label_struct).item() + 1, 1))
test_types_indices_list = [x[1].tolist() for x in kfold_obj.split(domain_types)]
test_types_list = [[domain_types[x] for x in z] for z in test_types_indices_list]
test_types_list = test_types_list[:1]

split_nrs = range(len(test_types_list))
GPU_DEVICE = 0

for split_nr, test_types in zip(split_nrs, test_types_list):
    train_types = list(set(range(1, np.prod(domain_label_struct).item() + 1, 1)) - set(test_types))
    train_indices, test_indices = \
        np.where(np.isin(domain_labels, test_elements=np.asarray(train_types)))[0], \
        np.where(np.isin(domain_labels, test_elements=np.asarray(test_types)))[0]

    file_path = 'tmp/' + uuid.uuid4().hex + '.pickle'
    with open(file_path, 'wb') as handle:
        pickle.dump(
            obj={'train_indices': train_indices, 'test_indices': test_indices, 'train_types': train_types},
            file=handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Streams/{}_{}_{}_stdout.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stdoutFile, \
         open('Streams/{}_{}_{}_stderr.txt'.format(args.model_name, CUR_DOMAIN_FACTOR_NAME, str(split_nr)), 'w') as stderrFile:

        run([
            'python', f_name,
            '-e_s', '30',
            '-b_s', '16',
            '-s', '42',
            '-d_t', args.data_type,
            '-do_t', args.domain_type,
            '-d', args.dataset,
            '-cv_s', str(split_nr),
            '-g', str(GPU_DEVICE),
            '-d_f_p', dataset_path,
            '-f_p', file_path,
            '-m_n', args.model_name,
        ], stdout=stdoutFile, stderr=stderrFile)
