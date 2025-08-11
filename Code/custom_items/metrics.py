from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked
import tensorflow_addons as tfa
from typing import Optional
import tensorflow as tf


class MultiClassPrecision(tfa.metrics.FBetaScore):
    @typechecked
    def __init__(self, num_classes: FloatTensorLike, average: str = None, threshold: Optional[FloatTensorLike] = None,
            name: str = "precision", dtype: AcceptableDTypes = None, ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(self.weights_intermediate, tf.reduce_sum(self.weights_intermediate))
            precision = tf.reduce_sum(precision * weights)

        elif self.average is not None:  # [micro, macro]
            precision = tf.reduce_mean(precision)

        return precision

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


class MultiClassRecall(tfa.metrics.FBetaScore):
    @typechecked
    def __init__(self, num_classes: FloatTensorLike, average: str = None, threshold: Optional[FloatTensorLike] = None,
            name: str = "recall", dtype: AcceptableDTypes = None, ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def result(self):
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(self.weights_intermediate, tf.reduce_sum(self.weights_intermediate))
            recall = tf.reduce_sum(recall * weights)

        elif self.average is not None:  # [micro, macro]
            recall = tf.reduce_mean(recall)

        return recall

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
