from custom_items.data_fetching import fetch_labels_indices, dataset_constructor, dataset_constructor_signfi
from custom_items.callbacks import WeightRestoreCallback
from custom_items.pretrain_models import PretrainModel
from custom_items.losses import nt_xent_loss
from models.backbone import MobileNetV2Backbone
import tensorflow_addons as tfa
import keras_tuner as kt
import tensorflow as tf
import argparse
import pickle
import os


parser = argparse.ArgumentParser(description="Experiment automation setup script.")
parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment', required=True)
parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=True)
parser.add_argument('-s', '--seed', type=int, help='<Required> Random seed used for random ops in the experiment', required=True)
parser.add_argument('-l_r', '--learning_rate', type=float, help='<Required> Learning rate to be used in the experiment', required=True)
parser.add_argument('-t', '--tau', type=float, help='<Required> Tau (temperature) used for ntxent loss in the experiment', required=True)
# parser.add_argument('-d_l_w', '--domain_loss_weight', type=float, help='<Required> Weight used for effect of domain loss on overall loss in the experiment', required=True)
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
        domain_nr = 5
    elif args.dataset == 'signfi' and args.domain_type == 'environment':
        data_fetch_path = r'Datasets/signfi-domain-leave-out-dataset-raw-environment.hdf5'
        input_shape = (224, 64, 3)
        domain_nr = 2
    elif args.dataset == 'widar3':
        data_fetch_path = r'Datasets/widar3.0-domain-leave-out-dataset-benchmark-raw.hdf5'
        input_shape = (2048, 64, 6)
        domain_nr = 150
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

# ------------------------------- Pre-training with integrated domain classification -------------------------------


def model_builder(hp):
    inp = tf.keras.layers.Input(shape=input_shape)
    cnn_extractors = MobileNetV2Backbone(input_shape=input_shape, backbone_name=args.model_name + "_backbone").get_model()
    enc = cnn_extractors(inp)

    proj = tf.keras.layers.Dense(450, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(enc)
    proj = tf.keras.layers.Activation(tf.keras.activations.relu)(proj)
    proj = tf.keras.layers.Dense(300, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(proj)
    proj = tf.keras.layers.Activation(tf.keras.activations.relu)(proj)
    proj = tf.keras.layers.Dense(150, use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.33, mode='fan_in', distribution='uniform', seed=args.seed))(proj)

    pretrain_model = PretrainModel(inp, proj, name=args.model_name)
    pretrain_model.compile(optimizer=tfa.optimizers.SGDW(learning_rate=args.learning_rate * 5,
                                                         weight_decay=hp.Choice('weight_decay', values=[0.0, 1e-3, 1e-4, 1e-5, 1e-6]),
                                                         momentum=hp.Choice('momentum', values=[0.99, 0.95, 0.9, 0.85, 0.8]),
                                                         nesterov=hp.Boolean('nesterov', default=False)),
                           sim_loss_fn=nt_xent_loss, tau=args.tau)

    return pretrain_model


# ------------------------------------------------------------------------------------------------------------------

if args.dataset == 'widar3':
    construct_func = dataset_constructor
elif args.dataset == 'signfi':
    construct_func = dataset_constructor_signfi
else:
    raise ValueError("Unknown dataset: {}".format(args.dataset))

pre_train_set = construct_func(indices_types_dict["train_indices"][train_instance_indices],
                               data_fetch_path,
                               'pre-train', args.batch_size, args.seed, args.data_type, args.domain_type)
pre_val_set = construct_func(indices_types_dict["train_indices"][val_instance_indices],
                             data_fetch_path,
                             'pre-train-val', args.batch_size, args.seed, args.data_type, args.domain_type)

pre_callback_objects = [
    WeightRestoreCallback(monitor='val_loss', min_delta=0, restore_best_weights=True)
]

tuner = kt.tuners.bayesian.BayesianOptimization(
    hypermodel=model_builder,
    objective=kt.Objective('val_loss', 'min'),
    max_trials=20,
    overwrite=False
)

tuner.search(x=pre_train_set, epochs=args.epoch_size, steps_per_epoch=len(train_instance_indices) // args.batch_size,
             verbose=2, callbacks=pre_callback_objects, validation_data=pre_val_set,
             validation_steps=len(val_instance_indices) // args.batch_size, validation_freq=1)

best_hps = tuner.get_best_hyperparameters()[0]
print("Best task loss weight: {}, best domain loss weight: {}".format(best_hps.get('task_loss_weight'), best_hps.get('domain_loss_weight')))
