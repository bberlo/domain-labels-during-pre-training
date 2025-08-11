import tensorflow as tf


# Used for complete tuning procedure from scratch
class MobileNetV2Backbone:
    def __init__(self, hp, input_shape=(2048, 64, 6)):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='normal', seed=None)

        self.input_shape = input_shape
        self.hp = hp

        for shape_elem in input_shape[:-1]:
            if shape_elem % 2 != 0:
                raise ValueError("Input width or height should be divisible by 2.")

    def get_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape, name="raw_input")
        batch_norm = self.hp.Boolean('batch_norm', default=False)
        inp_filters = self.hp.Int('filters_inp_conv', min_value=self.input_shape[-1], max_value=36, step=2, default=12)

        x = tf.keras.layers.Conv2D(
            inp_filters,
            (self.hp.Int('kernel_size_inp_conv_i', min_value=2, max_value=36, step=2, default=24),
             self.hp.Int('kernel_size_inp_conv_j', min_value=2, max_value=36, step=2, default=24)),
            (1, 1),
            'same', activation=None, use_bias=False, kernel_initializer=self.kernel_initializer)(inp)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.MaxPooling2D(
            (self.hp.Int('pool_size_maxP_inp_conv_i', min_value=2, max_value=36, step=2, default=24),
             self.hp.Int('pool_size_maxP_inp_conv_j', min_value=2, max_value=36, step=2, default=24)),
            (2, 2),
            'same')(x)

        depth_1 = self.hp.Int('backbone_depth', min_value=2, max_value=5, step=1, default=3)
        backbone_params = []

        for i in range(depth_1):
            params_elem = [self.hp.Int('mbConv_' + str(i) + '_kernel_size', min_value=2, max_value=8, step=1, default=5)]

            if i == 0:
                params_elem.append(inp_filters)  # 'mbConv_' + str(i) + '_inp_filters'
            else:
                params_elem.append(backbone_params[i-1][1])

            params_elem.append(params_elem[-1] * 2)  # 'mbConv_' + str(i) + '_outp_filters'
            params_elem.append(self.hp.Int('mbConv_' + str(i) + '_exp_ratio', min_value=2, max_value=8, step=1, default=3))
            params_elem.append(self.hp.Boolean('mbConv_' + str(i) + '_id_skip', default=False))
            params_elem.append((1, 1))  # 'mbConv_' + str(i) + '_strides_i'
            params_elem.append(None)  # drop_rate
            params_elem.append(self.hp.Int('mbConv_' + str(i) + '_num_repeat', min_value=0, max_value=2, step=1, default=1))

            backbone_params.append(params_elem)

        for index, params in enumerate(backbone_params):
            x = self.mobilev2_block(
                kernel_size=params[0],
                inp_filters=params[1],
                outp_filters=params[2],
                exp_ratio=params[3],
                id_skip=params[4],
                strides=params[5],
                drop_rate=params[6],
                batch_norm=batch_norm
            )(x)

            for _ in range(params[7]):
                x = self.mobilev2_block(
                    kernel_size=params[0],
                    inp_filters=params[2],
                    outp_filters=params[2],
                    exp_ratio=params[3],
                    id_skip=params[4],
                    strides=(1, 1),
                    drop_rate=params[6],
                    batch_norm=batch_norm
                )(x)

            x = tf.keras.layers.MaxPooling2D(
                (self.hp.Int('pool_size_maxP_{}_i'.format(index), min_value=2, max_value=36, step=2, default=24),
                 self.hp.Int('pool_size_maxP_{}_j'.format(index), min_value=2, max_value=36, step=2, default=24)),
                (2, 2),
                'same')(x)

        x_interm = tf.keras.layers.Flatten()(x)

        return tf.keras.models.Model(inp, x_interm)

    def mobilev2_block(self, kernel_size, inp_filters, outp_filters, exp_ratio,
                       strides, id_skip=None, drop_rate=None, batch_norm=None):

            def block(inputs):
                res = inputs

                # Expansion
                x = tf.keras.layers.Conv2D(
                    filters=inp_filters * exp_ratio,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer)(inputs)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)
                x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

                # Depthwise convolution
                x = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    depthwise_initializer=self.kernel_initializer)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)
                x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

                # Output
                x = tf.keras.layers.Conv2D(
                    filters=outp_filters,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)

                if id_skip:

                    if x._type_spec.shape != res._type_spec.shape:

                        res = tf.keras.layers.Conv2D(
                            filters=x._type_spec.shape[-1],
                            kernel_size=(1, 1),
                            strides=strides,
                            padding='same',
                            activation=None,
                            use_bias=True,
                            kernel_initializer=self.kernel_initializer
                        )(res)

                        if batch_norm:
                            res = tf.keras.layers.BatchNormalization(axis=3)(res)

                    x = tf.keras.layers.Add()([x, res])

                if drop_rate:
                    x = tf.keras.layers.Dropout(drop_rate)(x)

                return x

            return block


# Can be used for tuning individual components
class MobileNetV2Backbone2:
    def __init__(self, hp, input_shape=(2048, 64, 6)):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='normal', seed=None)

        self.input_shape = input_shape
        self.hp = hp

    def get_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape, name="raw_input")

        x = tf.keras.layers.Conv2D(21, (12, 36), (1, 1), 'same', activation=None, use_bias=False, kernel_initializer=self.kernel_initializer)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.AvgPool2D((4, 16), (2, 2), 'same')(x)

        # ---------- Mobile block set 1 ----------
        x = self.mobilev2_block(
            kernel_size=4,
            inp_filters=21,
            outp_filters=42,
            exp_ratio=2,
            strides=1,
            batch_norm=True,
            id_skip=False
        )(x)
        x = self.mobilev2_block(
            kernel_size=4,
            inp_filters=42,
            outp_filters=42,
            exp_ratio=2,
            strides=1,
            batch_norm=True,
            id_skip=False
        )(x)
        x = self.mobilev2_block(
            kernel_size=4,
            inp_filters=42,
            outp_filters=42,
            exp_ratio=2,
            strides=1,
            batch_norm=True,
            id_skip=False
        )(x)
        x = tf.keras.layers.AvgPool2D((26, 20), (2, 2), 'same')(x)

        # ---------- Mobile block set 2 ----------
        x = self.mobilev2_block(
            kernel_size=8,
            inp_filters=42,
            outp_filters=84,
            exp_ratio=4,
            strides=1,
            batch_norm=True,
            id_skip=True
        )(x)
        x = self.mobilev2_block(
            kernel_size=8,
            inp_filters=84,
            outp_filters=84,
            exp_ratio=4,
            strides=1,
            batch_norm=True,
            id_skip=True
        )(x)
        x = tf.keras.layers.AvgPool2D((30, 22), (2, 2), 'same')(x)

        # ---------- Mobile block set 3 ----------
        x = self.mobilev2_block(
            kernel_size=5,
            inp_filters=84,
            outp_filters=168,
            exp_ratio=7,
            strides=1,
            batch_norm=True,
            id_skip=True
        )(x)
        x = tf.keras.layers.AvgPool2D((2, 6), (x._type_spec.shape[1] // 2, 4), 'same')(x)

        x_interm = tf.keras.layers.Flatten()(x)

        return tf.keras.models.Model(inp, x_interm)

    def mobilev2_block(self, kernel_size, inp_filters, outp_filters, exp_ratio,
                       strides, id_skip=None, drop_rate=None, batch_norm=None):

            def block(inputs):
                res = inputs

                # Expansion
                x = tf.keras.layers.Conv2D(
                    filters=inp_filters * exp_ratio,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer)(inputs)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)
                x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

                # Depthwise convolution
                x = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    depthwise_initializer=self.kernel_initializer)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)
                x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

                # Output
                x = tf.keras.layers.Conv2D(
                    filters=outp_filters,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    activation=None,
                    kernel_initializer=self.kernel_initializer)(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization(axis=3)(x)

                if id_skip:

                    if x._type_spec.shape != res._type_spec.shape:

                        res = tf.keras.layers.Conv2D(
                            filters=x._type_spec.shape[-1],
                            kernel_size=(1, 1),
                            strides=strides,
                            padding='same',
                            activation=None,
                            use_bias=True,
                            kernel_initializer=self.kernel_initializer
                        )(res)

                        if batch_norm:
                            res = tf.keras.layers.BatchNormalization(axis=3)(res)

                    x = tf.keras.layers.Add()([x, res])

                if drop_rate:
                    x = tf.keras.layers.Dropout(drop_rate)(x)

                return x

            return block
