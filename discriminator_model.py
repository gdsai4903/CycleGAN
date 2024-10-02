import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self):
        '''
        Initialize the Discriminator model with multiple downsample layers and convolutional layers.
        '''
        super(Discriminator, self).__init__()

        initializer = tf.random_normal_initializer(0., 0.02)

        # Define downsampling layers
        self.down1 = self.downsample(64, 4, apply_batchnorm=False)
        self.down2 = self.downsample(128, 4)
        self.down3 = self.downsample(256, 4)

        # Additional layers for final processing
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.leady_relu = tf.keras.layers.LeakyReLU()
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)


    def downsample(self, filters, size, apply_batchnorm=True):
        '''
        Downsample the input using Conv2D and BatchNormalization.
        
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
    
    def call(self, inputs, target, training=False):
        '''
        The forward pass of the Discriminator. Concatenates the input and target images
        and applies downsampling and convolutional layers to classify them.
        
        Args:
            inputs (tensor): The input image tensor.
            target (tensor): The target image tensor.
            training (bool): Indicates whether the model is in training mode.
        
        Returns:
            The final output after applying the layers.
        '''
        x = tf.keras.layers.concatenate([inputs, target])
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.zero_pad1(x)
        x = self.leady_relu(x)
        x = self.conv(x)
        x = self.batchnorm1(x, training=training)
        x = self.zero_pad2(x)
        return self.last(x)