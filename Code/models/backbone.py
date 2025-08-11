from custom_items.layers import ResNetBlock
import tensorflow as tf
import math


class Backbone:
    def __init__(self, backbone_name, input_shape=(128, 2048, 1)):
        self.kernel_initializer = "he_normal"
        self.input_shape = input_shape
        self.backbone_name = backbone_name

        for shape_elem in input_shape[:-1]:
            if shape_elem % 2 != 0:
                raise ValueError("Input width or height should be divisible by 2.")

    def get_model(self):
        inp_v1 = tf.keras.layers.Input(shape=self.input_shape, name="dfs_input_v1")
        inp_v2 = tf.keras.layers.Input(shape=self.input_shape, name="dfs_input_v2")

        # --------- ResNet18 for view 1 --------- #
        x_v1 = tf.keras.layers.Conv2D(64, (7, 7), 2, 'same', kernel_initializer=self.kernel_initializer)(inp_v1)
        x_v1 = tf.keras.layers.BatchNormalization()(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)

        x_v1 = tf.keras.layers.MaxPool2D((3, 3), 2, 'same')(x_v1)

        x_v1 = ResNetBlock(64, down_sample=False)(x_v1)
        x_v1 = ResNetBlock(64, down_sample=False)(x_v1)

        x_v1 = ResNetBlock(128, down_sample=True)(x_v1)
        x_v1 = ResNetBlock(128, down_sample=False)(x_v1)

        x_v1 = ResNetBlock(256, down_sample=True)(x_v1)
        x_v1 = ResNetBlock(256, down_sample=False)(x_v1)

        x_v1 = ResNetBlock(512, down_sample=True)(x_v1)
        x_v1 = ResNetBlock(512, down_sample=False)(x_v1)

        out_v1 = tf.keras.layers.GlobalAveragePooling2D()(x_v1)

        # --------- ResNet18 for view 2 --------- #
        x_v2 = tf.keras.layers.Conv2D(64, (7, 7), 2, 'same', kernel_initializer=self.kernel_initializer)(inp_v2)
        x_v2 = tf.keras.layers.BatchNormalization()(x_v2)
        x_v2 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v2)

        x_v2 = tf.keras.layers.MaxPool2D((3, 3), 2, 'same')(x_v2)

        x_v2 = ResNetBlock(64, down_sample=False)(x_v2)
        x_v2 = ResNetBlock(64, down_sample=False)(x_v2)

        x_v2 = ResNetBlock(128, down_sample=True)(x_v2)
        x_v2 = ResNetBlock(128, down_sample=False)(x_v2)

        x_v2 = ResNetBlock(256, down_sample=True)(x_v2)
        x_v2 = ResNetBlock(256, down_sample=False)(x_v2)

        x_v2 = ResNetBlock(512, down_sample=True)(x_v2)
        x_v2 = ResNetBlock(512, down_sample=False)(x_v2)

        out_v2 = tf.keras.layers.GlobalAveragePooling2D()(x_v2)

        return tf.keras.models.Model([inp_v1, inp_v2], [out_v1, out_v2], name=self.backbone_name)

    def get_model_one_stream(self):
        inp_v1 = tf.keras.layers.Input(shape=self.input_shape, name="dfs_input_v1")

        # --------- ResNet18 --------- #
        x_v1 = tf.keras.layers.Conv2D(64, (7, 7), 2, 'same', kernel_initializer=self.kernel_initializer)(inp_v1)
        x_v1 = tf.keras.layers.BatchNormalization()(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)

        x_v1 = tf.keras.layers.MaxPool2D((3, 3), 2, 'same')(x_v1)

        x_v1 = ResNetBlock(64, down_sample=False)(x_v1)
        x_v1 = ResNetBlock(64, down_sample=False)(x_v1)

        x_v1 = ResNetBlock(128, down_sample=True)(x_v1)
        x_v1 = ResNetBlock(128, down_sample=False)(x_v1)

        x_v1 = ResNetBlock(256, down_sample=True)(x_v1)
        x_v1 = ResNetBlock(256, down_sample=False)(x_v1)

        x_v1 = ResNetBlock(512, down_sample=True)(x_v1)
        x_v1 = ResNetBlock(512, down_sample=False)(x_v1)

        out_v1 = tf.keras.layers.GlobalAveragePooling2D()(x_v1)

        return tf.keras.models.Model(inp_v1, out_v1, name=self.backbone_name)


class BackboneShallow:
    def __init__(self, backbone_name, input_shape=(128, 2048, 1)):
        self.kernel_initializer = "he_uniform"
        self.input_shape = input_shape
        self.backbone_name = backbone_name

        for shape_elem in input_shape[:-1]:
            if shape_elem % 2 != 0:
                raise ValueError("Input width or height should be divisible by 2.")

    def get_model(self):
        inp_v1 = tf.keras.layers.Input(shape=self.input_shape, name="dfs_input_v1")
        # inp_v2 = tf.keras.layers.Input(shape=self.input_shape, name="dfs_input_v2")

        # --------- ResNet18 for view 1 --------- #
        x_v1 = tf.keras.layers.Conv2D(32, (24, 24), 1, 'same', kernel_initializer=self.kernel_initializer, use_bias=False)(inp_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)
        x_v1 = tf.keras.layers.MaxPool2D((4, 4), 2, 'same')(x_v1)

        x_v1 = tf.keras.layers.Conv2D(64, (16, 16), 1, 'same', kernel_initializer=self.kernel_initializer, use_bias=False)(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)
        x_v1 = tf.keras.layers.MaxPool2D((4, 4), 2, 'same')(x_v1)

        x_v1 = tf.keras.layers.Conv2D(96, (8, 8), 1, 'same', kernel_initializer=self.kernel_initializer, use_bias=False)(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)
        x_v1 = tf.keras.layers.MaxPool2D((4, 4), 2, 'same')(x_v1)

        x_v1 = tf.keras.layers.Conv2D(128, (4, 4), 1, 'same', kernel_initializer=self.kernel_initializer, use_bias=False)(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)
        x_v1 = tf.keras.layers.MaxPool2D((4, 4), 2, 'same')(x_v1)

        x_v1 = tf.keras.layers.Conv2D(160, (2, 2), 1, 'same', kernel_initializer=self.kernel_initializer, use_bias=False)(x_v1)
        x_v1 = tf.keras.layers.BatchNormalization()(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)
        x_v1 = tf.keras.layers.MaxPool2D((4, 4), 2, 'same')(x_v1)

        out_v1 = tf.keras.layers.GlobalMaxPooling2D()(x_v1)

        return tf.keras.models.Model(inp_v1, out_v1, name=self.backbone_name)


class SignFiBackbone:
    def __init__(self, backbone_name, input_shape=(224, 64, 3)):
        self.kernel_initializer = "he_normal"
        self.input_shape = input_shape
        self.backbone_name = backbone_name

    def get_model(self):
        inp_v1 = tf.keras.layers.Input(shape=self.input_shape, name="dfs_input_v1")

        x_v1 = tf.keras.layers.Cropping2D(cropping=(12, 2))(inp_v1)
        x_v1 = tf.keras.layers.ZeroPadding2D((1, 1))(x_v1)

        x_v1 = tf.keras.layers.Conv2D(3, (3, 3), 1, 'valid',
                                      kernel_initializer=self.kernel_initializer,
                                      use_bias=True,
                                      kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x_v1)

        x_v1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)(x_v1)
        x_v1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_v1)

        x_v1 = tf.keras.layers.AvgPool2D((3, 3), 3, 'valid')(x_v1)
        x_v1 = tf.keras.layers.Dropout(rate=0.6)(x_v1)
        out_v1 = tf.keras.layers.Flatten()(x_v1)

        return tf.keras.models.Model(inp_v1, out_v1, name=self.backbone_name)


class MobileNetV2Backbone:
    def __init__(self, backbone_name, input_shape=(224, 64, 3)):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_out', distribution='normal', seed=None)

        self.input_shape = input_shape
        self.backbone_name = backbone_name

        # Please note: interpolated function has been tested on time dimensions 224, 2048
        self.time_pool_value = math.floor(0.00109649 * input_shape[0] + 1.75439)

    def get_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape, name="raw_input")

        x = tf.keras.layers.Conv2D(21, (12, 36), (1, 1), 'same', activation=None, use_bias=False, kernel_initializer=self.kernel_initializer)(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.AvgPool2D((4, 16), (self.time_pool_value, 2), 'same')(x)

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
        x = tf.keras.layers.AvgPool2D((26, 20), (self.time_pool_value, 2), 'same')(x)

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
        x = tf.keras.layers.AvgPool2D((30, 22), (self.time_pool_value, 2), 'same')(x)

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

        return tf.keras.models.Model(inp, x_interm, name=self.backbone_name)

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
