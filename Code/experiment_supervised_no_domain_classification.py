from custom_items.data_fetching import dataset_constructor, dataset_constructor_signfi
from custom_items.metrics import MultiClassPrecision, MultiClassRecall
from custom_items.data_fetching import fetch_labels_indices
from custom_items.callbacks import WeightRestoreCallback
from models.backbone import MobileNetV2Backbone
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import argparse
import datetime
import pickle
import os


parser = argparse.ArgumentParser(description="Experiment automation setup script.")
parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment', required=True)
parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=True)
parser.add_argument('-s', '--seed', type=int, help='<Required> Random seed used for random ops in the experiment', required=True)
parser.add_argument('-l_r', '--learning_rate', type=float, help='<Required> Learning rate to be used in the experiment', required=True)
parser.add_argument('-d_t', '--data_type', help='<Required> Data type used during experiment: dfs, gaf', required=True)
parser.add_argument('-do_t', '--domain_type', help='<Required> Domain type: user, position, orientation', required=False)
parser.add_argument('-d', '--dataset', help='<Required> Dataset used during experiment: widar3, signfi', required=True)
parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
parser.add_argument('-d_f_n', '--domain_factor_name', help='<Required> Set domain factor name to be used in the experiment', required=True)
parser.add_argument('-f_p', '--file_path', help='<Required> File path to be used in the experiment', required=True)
args = parser.parse_args()

# GPU config. for allocating limited amount of memory on a given device
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        tf.config.experimental.set_visible_devices(gpus[args.gpu], "GPU")
    except RuntimeError as e:
        print(e)

# Set logging level
tf.get_logger().setLevel("WARNING")

# Load training, validation, and test data
with open(args.file_path, 'rb') as handle:
    indices_types_dict = pickle.load(handle)

if args.data_type == 'ampphase':
    if args.dataset == 'signfi' and args.domain_type == 'user':
        data_fetch_path = r'Datasets/signfi-domain-leave-out-dataset-raw-user.hdf5'
        input_shape = (224, 64, 3)
        class_nr = 150
    elif args.dataset == 'signfi' and args.domain_type == 'environment':
        data_fetch_path = r'Datasets/signfi-domain-leave-out-dataset-raw-environment.hdf5'
        input_shape = (224, 64, 3)
        class_nr = 276
    elif args.dataset == 'widar3':
        data_fetch_path = r'Datasets/widar3.0-domain-leave-out-dataset-benchmark-raw.hdf5'
        input_shape = (2048, 64, 12)
        class_nr = 6
    else:
        raise ValueError("Unknown dataset or domain type: {}, {}".format(args.dataset, args.domain_type))

else:
    raise ValueError("Unknown data_type. Allowed values: dfs, gaf.")

if args.dataset == 'widar3':
    train_instance_indices, val_instance_indices = fetch_labels_indices(f_path=data_fetch_path,
                                                                        indices=indices_types_dict["train_indices"],
                                                                        domain_types=indices_types_dict["train_types"],
                                                                        seed=args.seed)
elif args.dataset == 'signfi':
    train_instance_indices, val_instance_indices = fetch_labels_indices(f_path=data_fetch_path,
                                                                        indices=indices_types_dict["train_indices"],
                                                                        domain_types=indices_types_dict["train_types"],
                                                                        seed=args.seed, fine_tune=True)
else:
    raise ValueError("Unknown dataset: {}".format(args.dataset))

test_instances = indices_types_dict["test_indices"]
os.remove(args.file_path)

if args.dataset == 'widar3':
    construct_func = dataset_constructor
elif args.dataset == 'signfi':
    construct_func = dataset_constructor_signfi
else:
    raise ValueError("Unknown dataset: {}".format(args.dataset))

# ------------------------------------------------- End-to-end supervised training ----------------------------------
inp = tf.keras.layers.Input(shape=input_shape)
cnn_extractor = MobileNetV2Backbone(backbone_name=args.model_name + "_backbone_fine-tune", input_shape=input_shape).get_model()
enc = cnn_extractor(inp)

enc = tf.keras.layers.Dense(300, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(enc)
enc = tf.keras.layers.Activation(tf.keras.activations.relu)(enc)
enc = tf.keras.layers.Dense(class_nr, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(enc)
enc = tf.keras.layers.Activation(tf.keras.activations.softmax)(enc)

endtoend_model = tf.keras.models.Model(inp, enc, name=args.model_name + "_fine-tune")
endtoend_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=[tf.keras.metrics.CategoricalAccuracy(), MultiClassPrecision(num_classes=class_nr, average='weighted'),
                                MultiClassRecall(num_classes=class_nr, average='weighted'), tfa.metrics.F1Score(num_classes=class_nr, average='weighted'),
                                tfa.metrics.CohenKappa(num_classes=class_nr, sparse_labels=False)])
# -------------------------------------------------------------------------------------------------------------------

fine_train_set = construct_func(indices_types_dict["train_indices"][train_instance_indices],
                                data_fetch_path,
                                'pre-train', args.batch_size, args.seed, args.data_type, args.domain_type, True)
fine_val_set = construct_func(indices_types_dict["train_indices"][val_instance_indices],
                              data_fetch_path,
                              'pre-train-val', args.batch_size, args.seed, args.data_type, args.domain_type, True)

fine_callback_objects = [
    WeightRestoreCallback(monitor='val_loss', min_delta=0, restore_best_weights=True)
]

endtoend_train_history = endtoend_model.fit(x=fine_train_set, epochs=args.epoch_size, steps_per_epoch=len(train_instance_indices) // args.batch_size,
                                       verbose=2, callbacks=fine_callback_objects, validation_data=fine_val_set,
                                       validation_steps=len(val_instance_indices) // args.batch_size, validation_freq=1)
endtoend_history_frame = pd.DataFrame(endtoend_train_history.history)

test_set = construct_func(test_instances, data_fetch_path,
                          'test', args.batch_size, args.seed, args.data_type, args.domain_type, True)

_, acc, precision, recall, f_score, kappa_score = endtoend_model.evaluate(x=test_set, steps=len(test_instances) // args.batch_size, verbose=2)
metrics_frame = pd.DataFrame(data=[[acc, precision, recall, f_score, kappa_score]], columns=['A', 'P', 'R', 'F', 'CK'])

results_frame = pd.concat([endtoend_history_frame, metrics_frame], axis=1)
results_frame.to_csv("results/{}_results_dfn_{}_dt_{}_ds_{}_cvs_{}_{}.csv".format(args.model_name, args.domain_factor_name, args.data_type, args.dataset, str(args.crossval_split), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
