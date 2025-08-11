from tensorflow.python.keras.losses import LossFunctionWrapper
from .layers import RowInterlace
import tensorflow as tf


class PretrainModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, sim_loss_fn=None, metrics=None,
                loss_weights=None, tau=0.02, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        super(PretrainModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                       run_eagerly, steps_per_execution, **kwargs)

        self.tau = tau
        self.sim_loss_fn = sim_loss_fn
        self.interlace = RowInterlace()

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]

        with tf.GradientTape() as unsup_tape:

            pi_pred = self(x[0], training=True)
            pj_pred = self(x[1], training=True)

            pij_pred = self.interlace([pi_pred, pj_pred])
            sim_loss = self.sim_loss_fn(pij_pred, self.tau)

            if self.losses:
                sim_loss += tf.add_n(self.losses)

        trainable_vars = self.trainable_variables
        gradients = unsup_tape.gradient(sim_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = sim_loss

        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]

        pi_pred = self(x[0], training=False)
        pj_pred = self(x[1], training=False)

        pij_pred = self.interlace([pi_pred, pj_pred])
        sim_loss = self.sim_loss_fn(pij_pred, self.tau)

        if self.losses:
            sim_loss += tf.add_n(self.losses)

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = sim_loss

        return return_metrics


class DomPretrainModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, sim_loss_fn=None, metrics=None,
                loss_weights=None, tau=0.02, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, awareness_type=None, **kwargs):
        super(DomPretrainModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                       run_eagerly, steps_per_execution, **kwargs)

        self.tau = tau
        self.sim_loss_fn = sim_loss_fn
        self.interlace = RowInterlace()
        self.awareness_type = awareness_type

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]

        with tf.GradientTape() as unsup_tape:

            pi_pred = self(x[0], training=True)
            pj_pred = self(x[1], training=True)

            pij_pred = self.interlace([pi_pred, pj_pred])

            if self.awareness_type == 'filter_denom':

                if type(y) is tuple:
                    sij = self.interlace([y[0], y[1]])
                else:
                    sij = self.interlace([y, y])

                sim_loss = self.sim_loss_fn(pij_pred, sij, self.tau)
            elif self.awareness_type == 'batch':
                sim_loss = self.sim_loss_fn(pij_pred, self.tau)
            else:
                raise ValueError('Unknown awareness type. Allowed values: filter_denom, batch')

            if self.losses:
                sim_loss += tf.add_n(self.losses)

        trainable_vars = self.trainable_variables
        gradients = unsup_tape.gradient(sim_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = sim_loss

        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]

        pi_pred = self(x[0], training=False)
        pj_pred = self(x[1], training=False)

        pij_pred = self.interlace([pi_pred, pj_pred])

        if self.awareness_type == 'filter_denom':

            if type(y) is tuple:
                sij = self.interlace([y[0], y[1]])
            else:
                sij = self.interlace([y, y])

            sim_loss = self.sim_loss_fn(pij_pred, sij, self.tau)
        elif self.awareness_type == 'batch':
            sim_loss = self.sim_loss_fn(pij_pred, self.tau)
        else:
            raise ValueError('Unknown awareness type. Allowed values: filter_denom, batch')

        if self.losses:
            sim_loss += tf.add_n(self.losses)

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = sim_loss

        return return_metrics


class AdvClassPretrainModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, domain_class_loss_type=None, domain_class_loss_fn=None, sim_loss_fn=None, discriminator=None, metrics=None,
                loss_weights=None, tau=0.02, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        super(AdvClassPretrainModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                       run_eagerly, steps_per_execution, **kwargs)

        self.tau = tau
        self.domain_class_loss_type = domain_class_loss_type
        self.domain_class_loss_fn = LossFunctionWrapper(fn=domain_class_loss_fn, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.sim_loss_fn = sim_loss_fn
        self.interlace = RowInterlace()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.discriminator = discriminator

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]

        # Training the classification path
        with tf.GradientTape(persistent=True) as unsup_and_domain_class_tape:

            if self.domain_class_loss_type == 'multi-label':
                pi_pred, si_logit = self(x[0], training=True)
                pj_pred, sj_logit = self(x[1], training=True)

                pij_pred = self.interlace([pi_pred, pj_pred])
                s_pred = self.interlace([si_logit, sj_logit])
            else:
                pi_pred, ei_pred = self(x[0], training=True)
                pj_pred, ej_pred = self(x[1], training=True)

                pij_pred = self.interlace([pi_pred, pj_pred])
                eij_pred = self.concat([ei_pred, ej_pred])
                s_pred = self.discriminator(eij_pred, training=True)

            sim_loss = self.sim_loss_fn(pij_pred, self.tau)
            domain_class_loss = self.domain_class_loss_fn(y, s_pred)

            if self.compiled_loss._loss_weights:
                weighted_sim_loss = sim_loss * self.compiled_loss._loss_weights[0]
                weighted_domain_class_loss = domain_class_loss * self.compiled_loss._loss_weights[1]
            else:
                weighted_sim_loss = sim_loss
                weighted_domain_class_loss = domain_class_loss

            overall_loss = weighted_sim_loss + weighted_domain_class_loss

            if self.losses:
                overall_loss += tf.add_n(self.losses)

        if self.domain_class_loss_type == 'multi-label':
            trainable_vars = self.trainable_variables
            gradients = unsup_and_domain_class_tape.gradient(overall_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        else:
            # Gradients across two separate models:
            # https://stackoverflow.com/questions/69545716/compute-gradients-across-two-models
            trainable_disc_vars = self.discriminator.trainable_variables
            trainable_vars = self.trainable_variables

            disc_gradients = unsup_and_domain_class_tape.gradient(overall_loss, trainable_disc_vars)
            gradients = unsup_and_domain_class_tape.gradient(overall_loss, trainable_vars)

            self.optimizer.apply_gradients(zip(disc_gradients, trainable_disc_vars))
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = overall_loss
        return_metrics["ntxent_sim_loss"] = sim_loss
        return_metrics["domain_class_loss"] = domain_class_loss

        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_view1 = x[0], x_view2 = x[1]

        if self.domain_class_loss_type == 'multi-label':
            pi_pred, si_logit = self(x[0], training=False)
            pj_pred, sj_logit = self(x[1], training=False)

            pij_pred = self.interlace([pi_pred, pj_pred])
            s_pred = self.interlace([si_logit, sj_logit])
        else:
            pi_pred, ei_pred = self(x[0], training=False)
            pj_pred, ej_pred = self(x[1], training=False)

            pij_pred = self.interlace([pi_pred, pj_pred])
            eij_pred = self.concat([ei_pred, ej_pred])
            s_pred = self.discriminator(eij_pred, training=False)

        sim_loss = self.sim_loss_fn(pij_pred, self.tau)
        domain_class_loss = self.domain_class_loss_fn(y, s_pred)

        if self.compiled_loss._loss_weights:
            weighted_sim_loss = sim_loss * self.compiled_loss._loss_weights[0]
            weighted_domain_class_loss = domain_class_loss * self.compiled_loss._loss_weights[1]
        else:
            weighted_sim_loss = sim_loss
            weighted_domain_class_loss = domain_class_loss

        overall_loss = weighted_sim_loss + weighted_domain_class_loss

        if self.losses:
            overall_loss += tf.add_n(self.losses)

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return_metrics["loss"] = overall_loss
        return_metrics["ntxent_sim_loss"] = sim_loss
        return_metrics["domain_class_loss"] = domain_class_loss

        return return_metrics


class DomAltFlowModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, sim_loss_fn=None, class_loss_fn=None, metrics=None,
                loss_weights=None, tau=0.02, weighted_metrics=None, run_eagerly=None, steps_per_execution=None,
                awareness_type=None, dset_type=None, transfer=None, **kwargs):
        super(DomAltFlowModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                       run_eagerly, steps_per_execution, **kwargs)

        self.tau = tau
        self.sim_loss_fn = sim_loss_fn
        self.class_loss_fn = class_loss_fn
        self.interlace = RowInterlace()
        self.awareness_type = awareness_type
        self.dset_type = dset_type
        self.transfer = transfer

        # Averaging idea taken from Contrastive Learning of General-Purpose Audio Representations, Saeed et al.
        # https://ieeexplore.ieee.org/document/9413528
        self.average = tf.keras.layers.Average()

    def train_step(self, data):

        x_finetune, y_finetune, sample_weight_finetune = tf.keras.utils.unpack_x_y_sample_weight(data[0]) # x_view1 = x[0], x_view2 = x[1]
        x_pretrain, y_pretrain, sample_weight_pretrain = tf.keras.utils.unpack_x_y_sample_weight(data[1])

        # Training the classification path
        with tf.GradientTape() as class_tape:

            if self.dset_type == 'signfi':
                if not self.transfer:
                    y_pred, _ = self(x_finetune, training=True)
                else:
                    y_pred, _ = self(tf.concat([x_finetune, x_finetune], axis=-1), training=True)

            elif self.dset_type == 'widar3':
                if not self.transfer:
                    y_pred_1, _ = self(x_finetune[0], training=True)
                    y_pred_2, _ = self(x_finetune[1], training=True)
                    y_pred = self.average([y_pred_1, y_pred_2])
                else:
                    y_pred_1, _ = self(x_finetune[0][..., :tf.shape(x_finetune[0])[-1] // 2], training=True)
                    y_pred_2, _ = self(x_finetune[1][..., :tf.shape(x_finetune[1])[-1] // 2], training=True)
                    y_pred_3, _ = self(x_finetune[0][..., tf.shape(x_finetune[0])[-1] // 2:], training=True)
                    y_pred_4, _ = self(x_finetune[1][..., tf.shape(x_finetune[1])[-1] // 2:], training=True)
                    y_pred = self.average([y_pred_1, y_pred_2, y_pred_3, y_pred_4])
            else:
                raise ValueError("Encountered unknown dset type. Allowed values: signfi, widar3.")

            class_loss = self.class_loss_fn(y_finetune, y_pred)

        trainable_vars = self.trainable_variables

        class_gradients = class_tape.gradient(class_loss, trainable_vars)
        self.optimizer[0].apply_gradients(zip(class_gradients, trainable_vars))

        # Training the contrastive path
        with tf.GradientTape() as unsup_tape:

            _, pi_pred = self(x_pretrain[0], training=True)
            _, pj_pred = self(x_pretrain[1], training=True)

            pij_pred = self.interlace([pi_pred, pj_pred])

            if self.awareness_type == 'filter_denom':

                if type(y_pretrain) is tuple:
                    sij = self.interlace([y_pretrain[0], y_pretrain[1]])
                else:
                    sij = self.interlace([y_pretrain, y_pretrain])

                sim_loss = self.sim_loss_fn(pij_pred, sij, self.tau)
            elif self.awareness_type == 'batch':
                sim_loss = self.sim_loss_fn(pij_pred, self.tau)
            else:
                raise ValueError('Unknown awareness type. Allowed values: filter_denom, batch')

            if self.losses:
                sim_loss += tf.add_n(self.losses)

        unsup_gradients = unsup_tape.gradient(sim_loss, trainable_vars)
        self.optimizer[1].apply_gradients(zip(unsup_gradients, trainable_vars))

        return_metrics = {}

        return_metrics["loss"] = class_loss + sim_loss
        return_metrics["loss_class"] = class_loss
        return_metrics["loss_sim"] = sim_loss

        self.compiled_metrics.update_state((y_finetune, None), (y_pred, None), (sample_weight_finetune, None))

        for metric in self.metrics:

            result = metric.result()

            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return return_metrics

    def test_step(self, data):

        x_finetune, y_finetune, sample_weight_finetune = tf.keras.utils.unpack_x_y_sample_weight(data[0]) # x_view1 = x[0], x_view2 = x[1]
        x_pretrain, y_pretrain, sample_weight_pretrain = tf.keras.utils.unpack_x_y_sample_weight(data[1])

        if self.dset_type == 'signfi':
            if not self.transfer:
                y_pred, _ = self(x_finetune, training=False)
            else:
                y_pred, _ = self(tf.concat([x_finetune, x_finetune], axis=-1), training=False)

        elif self.dset_type == 'widar3':
            if not self.transfer:
                y_pred_1, _ = self(x_finetune[0], training=False)
                y_pred_2, _ = self(x_finetune[1], training=False)
                y_pred = self.average([y_pred_1, y_pred_2])
            else:
                y_pred_1, _ = self(x_finetune[0][..., :tf.shape(x_finetune[0])[-1] // 2], training=False)
                y_pred_2, _ = self(x_finetune[1][..., :tf.shape(x_finetune[1])[-1] // 2], training=False)
                y_pred_3, _ = self(x_finetune[0][..., tf.shape(x_finetune[0])[-1] // 2:], training=False)
                y_pred_4, _ = self(x_finetune[1][..., tf.shape(x_finetune[1])[-1] // 2:], training=False)
                y_pred = self.average([y_pred_1, y_pred_2, y_pred_3, y_pred_4])
        else:
            raise ValueError("Encountered unknown dset type. Allowed values: signfi, widar3.")

        class_loss = self.class_loss_fn(y_finetune, y_pred)

        _, pi_pred = self(x_pretrain[0], training=False)
        _, pj_pred = self(x_pretrain[1], training=False)

        pij_pred = self.interlace([pi_pred, pj_pred])

        if self.awareness_type == 'filter_denom':

            if type(y_pretrain) is tuple:
                sij = self.interlace([y_pretrain[0], y_pretrain[1]])
            else:
                sij = self.interlace([y_pretrain, y_pretrain])

            sim_loss = self.sim_loss_fn(pij_pred, sij, self.tau)
        elif self.awareness_type == 'batch':
            sim_loss = self.sim_loss_fn(pij_pred, self.tau)
        else:
            raise ValueError('Unknown awareness type. Allowed values: filter_denom, batch')

        return_metrics = {}

        return_metrics["loss"] = class_loss + sim_loss
        return_metrics["loss_class"] = class_loss
        return_metrics["loss_sim"] = sim_loss

        self.compiled_metrics.update_state((y_finetune, None), (y_pred, None), (sample_weight_finetune, None))

        for metric in self.metrics:

            result = metric.result()

            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return return_metrics
