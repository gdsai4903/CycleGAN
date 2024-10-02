import os
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import preprocess_image, postprocess_image
from tqdm import tqdm
from generator_model import Generator
from discriminator_model import Discriminator

class CycleGAN:
    def __init__(self, learning_rate=2e-4):
        '''
        Initialize the CycleGAN model with two generators and two discriminators,
        and define the optimizers and loss functions.
        
        Args:
            learning_rate (float): The learning rate for the Adam optimizers.
        '''
        # Initialize the generators and discriminators
        self.generator_g = Generator()  # X -> Y
        self.generator_f = Generator()  # Y -> X
        self.discriminator_x = Discriminator()  # X
        self.discriminator_y = Discriminator()  # Y

        # Define optimizers
        self.gen_g_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        self.gen_f_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        self.disc_x_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)
        self.disc_y_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.5)

        # Define the loss functions
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, real, generated):
        '''
        Calculate the discriminator loss for real and generated images.
        
        Args:
            real (tensor): The discriminator's prediction on real images.
            generated (tensor): The discriminator's prediction on generated images.
        
        Returns:
            Total loss for the discriminator.
        '''
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        '''
        Calculate the generator loss based on the discriminator's output.
        
        Args:
            generated (tensor): The discriminator's prediction on generated images.
        
        Returns:
            Loss for the generator.
        '''
        return self.loss_obj(tf.ones_like(generated), generated)

    def cycle_consistency_loss(self, real_image, cycled_image, lambda_cycle=10):
        '''
        Calculate the cycle consistency loss for image reconstruction.
        
        Args:
            real_image (tensor): The real image input.
            cycled_image (tensor): The image after a cycle of translation and reconstruction.
            lambda_cycle (int): The weight for the cycle consistency loss.
        
        Returns:
            Cycle consistency loss.
        '''
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return lambda_cycle * loss

    def identity_loss(self, real_image, same_image, lambda_identity=5):
        '''
        Calculate the identity loss for preserving image identity.
        
        Args:
            real_image (tensor): The real image input.
            same_image (tensor): The generated image that should look the same as the real one.
            lambda_identity (int): The weight for the identity loss.
        
        Returns:
            Identity loss.
        '''
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return lambda_identity * loss

    @tf.function
    def train_step(self, real_x, real_y):
        '''
        Perform a single training step, including forward and backward passes
        for both generators and discriminators.
        
        Args:
            real_x (tensor): Real images from domain X.
            real_y (tensor): Real images from domain Y.
        '''
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            # Generator F translates Y -> X
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # Discriminators evaluate real and generated images
            disc_real_x = self.discriminator_x(real_x, real_x, training=True)
            disc_fake_x = self.discriminator_x(fake_x, real_x, training=True)

            disc_real_y = self.discriminator_y(real_y, real_y, training=True)
            disc_fake_y = self.discriminator_y(fake_y, real_y, training=True)

            # Calculate the losses
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            cycle_loss_g = self.cycle_consistency_loss(real_x, cycled_x)
            cycle_loss_f = self.cycle_consistency_loss(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + cycle_loss_g
            total_gen_f_loss = gen_f_loss + cycle_loss_f

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Apply gradients
        gen_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        gen_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        disc_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        disc_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        self.gen_g_optimizer.apply_gradients(zip(gen_g_gradients, self.generator_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(gen_f_gradients, self.generator_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(disc_x_gradients, self.discriminator_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(disc_y_gradients, self.discriminator_y.trainable_variables))

    def load_weights(self, load_dir):
        '''
        Load the saved model weights for both generators and discriminators.

        Args:
            load_dir (str): The directory from which to load the weights.
        '''
        # Create dummy input to "build" the models
        dummy_input = tf.random.normal([1, 256, 256, 3])
        dummy_target = tf.random.normal([1, 256, 256, 3])

        # Call the models with the dummy input to initialize variables
        self.generator_g(dummy_input)
        self.generator_f(dummy_input)
        self.discriminator_x(dummy_input, dummy_target)
        self.discriminator_y(dummy_input, dummy_target)

        # Create a directory for the current epoch if it doesn't exist
        load_dir = os.path.join(load_dir)
        os.makedirs(load_dir, exist_ok=True)
        
        # Load the saved weights for the models
        self.generator_g.load_weights(os.path.join(load_dir, 'generator_g.h5'))
        self.generator_f.load_weights(os.path.join(load_dir, 'generator_f.h5'))
        self.discriminator_x.load_weights(os.path.join(load_dir, 'discriminator_x.h5'))
        self.discriminator_y.load_weights(os.path.join(load_dir, 'discriminator_y.h5'))

        print(f"Weights loaded from {load_dir}")


    def save_weights(self, save_dir):
        '''
        Save the current model weights for both generators and discriminators.

        Args:
            save_dir (str): The directory where the weights will be saved.
        '''
        # Create a directory for the current epoch if it doesn't exist
        save_dir = os.path.join(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Save weights for both generators and discriminators in the new directory
        self.generator_g.save_weights(os.path.join(save_dir, 'generator_g.h5'))
        self.generator_f.save_weights(os.path.join(save_dir, 'generator_f.h5'))
        self.discriminator_x.save_weights(os.path.join(save_dir, 'discriminator_x.h5'))
        self.discriminator_y.save_weights(os.path.join(save_dir, 'discriminator_y.h5'))

        print(f"Weights saved.\n")


    def train(self, dataset_x, dataset_y, epochs, load_from=None, save_to=None):
        '''
        Train the CycleGAN model using the given datasets for a number of epochs.

        Args:
            dataset_x (tf.data.Dataset): The dataset for domain X (real images).
            dataset_y (tf.data.Dataset): The dataset for domain Y (target images).
            epochs (int): The number of training epochs.
            load_from (str, optional): Directory from which to load weights before training.
            save_to (str, optional): Directory where weights will be saved after training.
        '''
        # Loads weights if directory is provided
        if load_from:
            self.load_weights(load_from)

        # loop through the epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Use tqdm to display a progress bar for the batches
            with tqdm(total=len(dataset_x), desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
                for real_x, real_y in zip(dataset_x, dataset_y):
                    self.train_step(real_x, real_y)

                    # Update the progress bar after each batch
                    pbar.update(1)

            print(f"Epoch {epoch + 1} completed.")

        # Save weights if a save directory is provided
        if save_to:
            self.save_weights(save_to)             


    def transform_to_ocean(self, image_path):
        '''
        Transform the input image to the ocean domain using generator G.

        Args:
            image_path (str): Path to the input image.

        Returns:
            The generated image in the ocean domain.
        '''
        image = preprocess_image(image_path)
        trans_image = self.generator_g(image, training=False)

        # Post-process the generated image
        output_image = postprocess_image(trans_image)

        # Display the generated image
        plt.imshow(output_image.numpy())
        plt.axis('off')
        plt.show()

        return output_image
        

    def transform_to_normal(self, image_path):
        '''
        Transform the input ocean domain image back to the normal domain using generator F.

        Args:
            image_path (str): Path to the input image.

        Returns:
            The generated image in the normal domain.
        '''
        image = preprocess_image(image_path)
        trans_image = self.generator_f(image, training=False)

        # Post-process the generated image
        output_image = postprocess_image(trans_image)

        # Display the generated image
        plt.imshow(output_image.numpy())
        plt.axis('off')
        plt.show()

        return output_image
