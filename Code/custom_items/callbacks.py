from scipy.stats import linregress
import tensorflow as tf
import numpy as np
import math


# Callback to halt training when loss is negative or diverges (EarlyStopping doesn't account for this)
class HaltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.0 or logs.get('val_loss') < 0.0 or math.isnan(logs.get('loss')) or math.isnan(
                logs.get('val_loss')) or logs.get('loss') > 100000.0 or logs.get('val_loss') > 100000.0:
            self.model.stop_training = True
            logs['val_ntxent_sim_loss'] = 1000.0
            logs['val_domain_class_loss'] = 0.0


# Callback to allow every pre-train model to finish an equal amount of epochs
class WeightRestoreCallback(tf.keras.callbacks.EarlyStopping):

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)

        if current is None:
            return

        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if tf.__version__ < "2.5.0":
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.best_epoch = epoch

                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

        else:
            if self._is_improvement(current, self.best):
                self.best = current
                self.best_epoch = epoch

                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


# Gradually scale up d_l_w until proper divergence threshold is established
class DomainLossWeightWarmup(tf.keras.callbacks.Callback):

    def __init__(self, slope, max_epochs, ringbuf_size, verbose=0, domain_loss_weight=0.0, slope_bound_scale=10):
        super().__init__()
        self.max_epochs = max_epochs
        self.ringbuf_size = ringbuf_size
        self.desired_slope = slope
        self.slope_bound = self.desired_slope * slope_bound_scale
        self.domain_loss_values = RingBuffer(size_max=self.ringbuf_size)
        self.verbose = verbose
        self.wait = 0
        self.best_weights = None
        self.highest_loss_weight = np.array(domain_loss_weight)

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_begin(self, epoch, logs=None):

        if 5 < epoch < self.max_epochs:

            y_values = np.array(self.domain_loss_values.get())
            x_values = np.arange(start=0, stop=y_values.shape[0])
            curr_slope, _, _, _, _ = linregress(x_values, y_values)

            hit_max_bound = np.any(np.isclose(
                np.array(-self.log(0.0000001, order=100)),
                y_values, rtol=0.05, atol=0.001))

            large_diff = np.ptp(y_values) > 1.0

            if self.wait > 0:
                self.wait -= 1
                return

            if not self.desired_slope < curr_slope < self.slope_bound or (hit_max_bound or large_diff):

                self.model.set_weights(self.best_weights)

                if curr_slope < self.desired_slope and not (hit_max_bound or large_diff):
                    multipl = 10.0
                elif curr_slope > self.slope_bound and not (hit_max_bound or large_diff):
                    multipl = 0.1
                else:
                    multipl = 0.1

                new_loss_weight = self.log_scale_mult(
                    tf.keras.backend.get_value(self.model.compiled_loss._loss_weights[0]), 1.0)
                new_loss_weight_2 = self.log_scale_mult(
                    tf.keras.backend.get_value(self.model.compiled_loss._loss_weights[1]), multipl)

                if multipl == 10.0 and np.array_equal(new_loss_weight_2, self.highest_loss_weight):
                    return

                tf.keras.backend.set_value(self.model.compiled_loss._loss_weights[0], new_loss_weight)
                tf.keras.backend.set_value(self.model.compiled_loss._loss_weights[1], new_loss_weight_2)

                self.wait = self.ringbuf_size

                if multipl == 10.0:
                    self.highest_loss_weight = new_loss_weight_2

                if self.verbose > 0:
                    print('\nEpoch %d: DomainLossWeightWarmup increasing d_l_w '
                          'to %s.' % (epoch, float(new_loss_weight_2)))

            else:

                self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):

        if logs.get('domain_class_loss') is None:
            raise ValueError("'domain_class_loss' cannot be found. "
                             "Either include it in the training procedure"
                             "or rename discriminator loss.")

        self.domain_loss_values.append(logs.get('domain_class_loss'))

    @staticmethod
    def log_scale_mult(weight, multiplier):
        return weight * multiplier

    @staticmethod
    def log(x, order=2):
        s = 0

        for i in range(1, order):
            s += ((-1) ** (i + 1)) * ((x - 1) ** i) / i

        return s


# Taken from: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
class RingBuffer:

    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max, data=None):
        self.max = size_max

        if data is None:
            self.data = []
        else:
            self.data = data

    class __Full:

        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max

        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self, x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data
