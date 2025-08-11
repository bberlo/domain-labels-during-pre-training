import tensorflow as tf


class FinetuneModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
                weighted_metrics=None, run_eagerly=None, steps_per_execution=None, dset_type=None, transfer=False, end_to_end=False, **kwargs):
        super(FinetuneModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                           run_eagerly, steps_per_execution, **kwargs)

        self.dset_type = dset_type
        self.transfer = transfer
        self.end_to_end = end_to_end

        # Averaging idea taken from Contrastive Learning of General-Purpose Audio Representations, Saeed et al.
        # https://ieeexplore.ieee.org/document/9413528
        self.average = tf.keras.layers.Average()

    def train_step(self, data):

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:

            if self.dset_type == 'signfi':
                if not self.transfer:
                    y_pred = self(x, training=True)
                else:
                    if not self.end_to_end:
                        y_pred = self(tf.concat([x, x], axis=-1), training=True)
                    else:
                        y_pred = self(tf.concat([x, x, x, x], axis=-1), training=True)

            elif self.dset_type == 'widar3':
                if not self.transfer:
                    y_pred_1 = self(x[0], training=True)
                    y_pred_2 = self(x[1], training=True)
                    y_pred = self.average([y_pred_1, y_pred_2])
                else:
                    y_pred_1 = self(x[0][..., :tf.shape(x[0])[-1] // 2], training=True)
                    y_pred_2 = self(x[1][..., :tf.shape(x[1])[-1] // 2], training=True)
                    y_pred_3 = self(x[0][..., tf.shape(x[0])[-1] // 2:], training=True)
                    y_pred_4 = self(x[1][..., tf.shape(x[1])[-1] // 2:], training=True)
                    y_pred = self.average([y_pred_1, y_pred_2, y_pred_3, y_pred_4])
            else:
                raise ValueError("Encountered unknown dset type. Allowed values: signfi, widar3.")

            loss = self.compute_loss(x, y, y_pred, sample_weight)

        self._validate_target_and_loss(y, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        if self.dset_type == 'signfi':
            if not self.transfer:
                y_pred = self(x, training=False)
            else:
                y_pred = self(tf.concat([x, x], axis=-1), training=False)

        elif self.dset_type == 'widar3':
            if not self.transfer:
                y_pred_1 = self(x[0], training=False)
                y_pred_2 = self(x[1], training=False)
                y_pred = self.average([y_pred_1, y_pred_2])
            else:
                y_pred_1 = self(x[0][..., :tf.shape(x[0])[-1] // 2], training=False)
                y_pred_2 = self(x[1][..., :tf.shape(x[1])[-1] // 2], training=False)
                y_pred_3 = self(x[0][..., tf.shape(x[0])[-1] // 2:], training=False)
                y_pred_4 = self(x[1][..., tf.shape(x[1])[-1] // 2:], training=False)
                y_pred = self.average([y_pred_1, y_pred_2, y_pred_3, y_pred_4])
        else:
            raise ValueError("Encountered unknown dset type. Allowed values: signfi, widar3.")

        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)
