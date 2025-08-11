from custom_items.metrics import MultiClassPrecision, MultiClassRecall
from kerastuner_tensorboard_logger import TensorBoardLogger, setup_tb
from custom_items.data_fetching import fetch_labels_indices
from custom_items.tuners import HyperbandSizeFiltering, BayesianSizeFiltering
from multiprocess import set_start_method
from models import backbone_tuning
import tensorflow_addons as tfa
import tensorflow as tf
import argparse
import pickle
import os

# ---------- STD TRAINING WITH DFS DATA -------------------------------------------------------------------------


def build_model(hp):
    dense_initializers = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_out', distribution='uniform')

    # Note: use backbone_tuning.MobileNetV2Backbone for tuning entire architecture,
    # use backbone_tuning.MobileNetV2Backbone2 for tuning specific layer/hyperparams.

    std_extractor = backbone_tuning.MobileNetV2Backbone2(hp, input_shape=(224, 64, 3)).get_model()
    inp = tf.keras.layers.Input(shape=(224, 64, 3))
    enc_o = std_extractor(inp)

    layer_nr = hp.Int('classifier_depth', min_value=1, max_value=2, step=1)
    for layer in range(layer_nr):
        enc_o = tf.keras.layers.Dense(hp.Int('neurons classifier layer {}'.format(layer), min_value=16, max_value=1024,
                                             step=16), activation='relu', kernel_initializer=dense_initializers)(enc_o)
    outp = tf.keras.layers.Dense(150, activation='softmax', kernel_initializer=dense_initializers)(enc_o)

    complete_model = tf.keras.models.Model(inp, outp)
    complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                           metrics=[tf.keras.metrics.CategoricalAccuracy(), MultiClassPrecision(num_classes=150, average='weighted'),
                                    MultiClassRecall(num_classes=150, average='weighted'), tfa.metrics.F1Score(num_classes=150, average='weighted'),
                                    tfa.metrics.CohenKappa(num_classes=150, sparse_labels=False)])

    return complete_model


# ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment automation setup script.")
    parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment', required=True)
    parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment', required=True)
    parser.add_argument('-cv_s', '--crossval_split', type=int, help='<Required> Current cross val. split used in the experiment', required=True)
    parser.add_argument('-s', '--seed', type=int, help='<Required> Random seed used for random ops in the experiment', required=True)
    parser.add_argument('-d_t', '--data_type', help='<Required> Data type used during experiment: dfs, gaf', required=True)
    parser.add_argument('-do_t', '--domain_type', help='<Required> Domain type: user, position, orientation', required=False)
    parser.add_argument('-d', '--dataset', help='<Required> Dataset used during experiment: widar3, signfi', required=True)
    parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment', required=True)
    parser.add_argument('-f_p', '--file_path', help='<Required> File path to be used in the experiment', required=True)
    parser.add_argument('-d_f_p', '--data_file_path', help='<Required> Data file path to be used in the experiment', required=True)
    args = parser.parse_args()

    # Prevent main process from clogging up GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as e:
            print(e)

    # Load training, validation, and test data
    with open(args.file_path, 'rb') as handle:
        indices_types_dict = pickle.load(handle)

    # Only SignFi supported
    data_fetch_path = r'Datasets/signfi-domain-leave-out-dataset-raw-user.hdf5'
    input_shape = (224, 64, 3)
    class_nr = 150

    train_instance_indices, val_instance_indices = fetch_labels_indices(f_path=data_fetch_path,
                                                                        indices=indices_types_dict["train_indices"],
                                                                        domain_types=indices_types_dict["train_types"],
                                                                        seed=args.seed, fine_tune=True)

    test_instances = indices_types_dict["test_indices"]
    os.remove(args.file_path)

    set_start_method("spawn")
    
    # Note: use hyperband tuner for tuning entire architecture (backbone_tuning.MobileNetV2Backbone), 
    # use bayesian tuner for tuning specific layer/hyperparams. (backbone_tuning.MobileNetV2Backbone2)

    # tuner = HyperbandSizeFiltering(
    #     hypermodel=build_model,
    #     objective='val_loss',
    #     max_epochs=50,
    #     factor=3,
    #     hyperband_iterations=1,
    #     seed=42,
    #     directory=r'Streams',
    #     project_name=args.model_name,
    #     logger=TensorBoardLogger(metrics=["val_loss"], logdir='Streams/{}-hparams'.format(args.model_name))
    # )
    tuner = BayesianSizeFiltering(
        hypermodel=build_model,
        objective='val_categorical_accuracy',
        max_trials=20,
        seed=42,
        directory=r'Streams',
        project_name=args.model_name,
        logger=TensorBoardLogger(metrics=["val_loss"], logdir='Streams/{}-hparams'.format(args.model_name))
    )
    setup_tb(tuner)
    tuner.search(x=indices_types_dict["train_indices"][train_instance_indices], epochs=args.epoch_size, steps_per_epoch=len(indices_types_dict["train_indices"][train_instance_indices])//args.batch_size, verbose=2, gpu=args.gpu,
                 validation_data=indices_types_dict["train_indices"][val_instance_indices], validation_steps=len(indices_types_dict["train_indices"][val_instance_indices])//args.batch_size,
                 validation_freq=1, batch_size=args.batch_size, dataset_filepath=args.data_file_path,
                 seed=args.seed, data_type=args.data_type, domain_type=args.domain_type)
