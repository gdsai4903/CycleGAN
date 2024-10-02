import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self):
        '''
        Initialize the Generator model with downsampling and upsampling layers.
        '''
        super(Generator, self).__init__()

        # Downsampling stack
        self.down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
        ]

        # Upsampling stack
        self.up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
        ]

        # Last layer
        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(3,
                                                    4,
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    activation='tanh')

    
    def downsample(self, filters, size, apply_batchnorm=True):
        '''
        Downsample using Conv2D and optionally apply BatchNormalization.
        
        Args:
            filters (int): Number of filters in the Conv2D layer.
            size (int): Size of the kernel in the Conv2D layer.
            apply_batchnorm (bool): Whether to apply batch normalization.
        
        Returns:
            A Sequential model containing the downsampling layers.
        '''
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters,
                                          size,
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result
    
    def upsample(self, filters, size, apply_dropout=False):
        '''
        Upsample using Conv2DTranspose and optionally apply Dropout.
        
        Args:
            filters (int): Number of filters in the Conv2DTranspose layer.
            size (int): Size of the kernel in the Conv2DTranspose layer.
            apply_dropout (bool): Whether to apply dropout.
        
        Returns:
            A Sequential model containing the upsampling layers.
        '''
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters,
                                                   size,
                                                   strides=2,
                                                   padding='same', kernel_initializer=initializer,
                                                   use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result
    
    def call(self, inputs, training=False):
        '''
        The forward pass of the Generator model.
        
        Args:
            inputs (tensor): The input image tensor.
            training (bool): Indicates whether the model is in training mode.
        
        Returns:
            The generated image after applying the layers.
        '''
        x = inputs
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        return self.last(x)