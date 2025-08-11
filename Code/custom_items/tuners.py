from custom_items.utilities import keras_model_memory_usage_in_bytes
from custom_items.data_fetching import dataset_constructor_signfi
from custom_items.callbacks import HaltCallback
import kerastuner.engine.trial as trial_module
from multiprocess import Queue, Process
import tensorflow as tf
import kerastuner as kt
import math


# To access best hyperparameters and train from scratch see https://github.com/keras-team/keras-tuner/issues/41
class HyperbandSizeFiltering(kt.tuners.hyperband.Hyperband):
    def __init__(self, hypermodel, objective, max_epochs, factor, hyperband_iterations,
                 seed, directory, project_name, logger, distribution_strategy=None):

        self.queue = Queue()
        self.max_model_size_in_bytes = 4000000000

        super(HyperbandSizeFiltering, self).__init__(
            hypermodel=hypermodel,
            objective=objective,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            distribution_strategy=distribution_strategy,
            directory=directory,
            project_name=project_name,
            logger=logger
        )

    def on_trial_end(self, trial):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        if not trial.get_state().get("status") == trial_module.TrialStatus.INVALID:
            self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.COMPLETED)

        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        p = Process(target=self._build_and_fit_model_worker,
                    args=(self.hypermodel, fit_args, fit_kwargs, trial.hyperparameters,
                          self.queue, self.max_model_size_in_bytes))
        p.start()
        ret = self.queue.get()
        p.join()

        if isinstance(ret, str):
            self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.INVALID)

            dummy_history_obj = tf.keras.callbacks.History()
            dummy_history_obj.history.setdefault('val_loss', []).append(2.5)
            return dummy_history_obj
        else:
            history_obj = tf.keras.callbacks.History()
            history_obj.history = ret
            return history_obj

    @staticmethod
    def _build_and_fit_model_worker(hypermodel, fit_args, fit_kwargs, hyperparams, queue, max_bytes):
        # OP placement analysis for discovering throughput bottlenecks
        # tf.debugging.set_log_device_placement(True)

        # GPU config. for allocating limited amount of memory on a given device
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[fit_kwargs["gpu"]], True)
                tf.config.experimental.set_visible_devices(gpus[fit_kwargs["gpu"]], "GPU")
            except RuntimeError as e:
                print(e)

        # Set logging level
        tf.get_logger().setLevel("WARNING")

        fit_kwargs["callbacks"].extend([
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True),
            HaltCallback()
        ])

        fit_kwargs["x"] = dataset_constructor_signfi(fit_kwargs["x"], fit_kwargs["dataset_filepath"], 'pre-train',
                                                       fit_kwargs["batch_size"], fit_kwargs["seed"], fit_kwargs["data_type"],
                                                       fit_kwargs["domain_type"], True)

        fit_kwargs["validation_data"] = dataset_constructor_signfi(fit_kwargs["validation_data"], fit_kwargs["dataset_filepath"], 'pre-train-val',
                                                                     fit_kwargs["batch_size"], fit_kwargs["seed"], fit_kwargs["data_type"],
                                                                     fit_kwargs["domain_type"], True)

        model = hypermodel.build(hyperparams)
        model_size_in_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=fit_kwargs["batch_size"])
        print("Considering model with size: {}".format(model_size_in_bytes))

        del fit_kwargs["dataset_filepath"]
        del fit_kwargs["batch_size"]
        del fit_kwargs["gpu"]
        del fit_kwargs["seed"]
        del fit_kwargs["data_type"]
        del fit_kwargs["domain_type"]

        if model_size_in_bytes > max_bytes:
            queue.put('invalid')
        else:
            try:
                history = model.fit(*fit_args, **fit_kwargs)
                if history.history["loss"][-1] < 0.0 or history.history["val_loss"][-1] < 0.0 \
                        or math.isnan(history.history["loss"][-1]) or math.isnan(history.history["val_loss"][-1]):
                    queue.put('invalid')
                else:
                    queue.put(history.history)
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError):
                queue.put('invalid')


class BayesianSizeFiltering(kt.tuners.bayesian.BayesianOptimization):
    def __init__(self, hypermodel, objective, max_trials, seed, directory,
                 project_name, logger, distribution_strategy=None):

        self.queue = Queue()
        self.max_model_size_in_bytes = 4000000000

        super(BayesianSizeFiltering, self).__init__(
            hypermodel=hypermodel,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            distribution_strategy=distribution_strategy,
            directory=directory,
            project_name=project_name,
            logger=logger
        )

    def on_trial_end(self, trial):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        if not trial.get_state().get("status") == trial_module.TrialStatus.INVALID:
            self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.COMPLETED)

        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        p = Process(target=self._build_and_fit_model_worker,
                    args=(self.hypermodel, fit_args, fit_kwargs, trial.hyperparameters,
                          self.queue, self.max_model_size_in_bytes))
        p.start()
        ret = self.queue.get()
        p.join()

        if isinstance(ret, str):
            self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.INVALID)

            dummy_history_obj = tf.keras.callbacks.History()
            dummy_history_obj.history.setdefault('val_loss', []).append(2.5)
            return dummy_history_obj
        else:
            history_obj = tf.keras.callbacks.History()
            history_obj.history = ret
            return history_obj

    @staticmethod
    def _build_and_fit_model_worker(hypermodel, fit_args, fit_kwargs, hyperparams, queue, max_bytes):
        # OP placement analysis for discovering throughput bottlenecks
        # tf.debugging.set_log_device_placement(True)

        # GPU config. for allocating limited amount of memory on a given device
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[fit_kwargs["gpu"]], True)
                tf.config.experimental.set_visible_devices(gpus[fit_kwargs["gpu"]], "GPU")
            except RuntimeError as e:
                print(e)

        # Set logging level
        tf.get_logger().setLevel("WARNING")

        fit_kwargs["callbacks"].extend([
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True),
            HaltCallback()
        ])

        fit_kwargs["x"] = dataset_constructor_signfi(fit_kwargs["x"], fit_kwargs["dataset_filepath"], 'pre-train',
                                                       fit_kwargs["batch_size"], fit_kwargs["seed"], fit_kwargs["data_type"],
                                                       fit_kwargs["domain_type"], True)

        fit_kwargs["validation_data"] = dataset_constructor_signfi(fit_kwargs["validation_data"], fit_kwargs["dataset_filepath"], 'pre-train-val',
                                                                     fit_kwargs["batch_size"], fit_kwargs["seed"], fit_kwargs["data_type"],
                                                                     fit_kwargs["domain_type"], True)

        model = hypermodel.build(hyperparams)
        model_size_in_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=fit_kwargs["batch_size"])
        print("Considering model with size: {}".format(model_size_in_bytes))

        del fit_kwargs["dataset_filepath"]
        del fit_kwargs["batch_size"]
        del fit_kwargs["gpu"]
        del fit_kwargs["seed"]
        del fit_kwargs["data_type"]
        del fit_kwargs["domain_type"]

        if model_size_in_bytes > max_bytes:
            queue.put('invalid')
        else:
            try:
                history = model.fit(*fit_args, **fit_kwargs)
                if history.history["loss"][-1] < 0.0 or history.history["val_loss"][-1] < 0.0 \
                        or math.isnan(history.history["loss"][-1]) or math.isnan(history.history["val_loss"][-1]):
                    queue.put('invalid')
                else:
                    queue.put(history.history)
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError):
                queue.put('invalid')

