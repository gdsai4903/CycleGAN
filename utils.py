import tensorflow as tf

from PIL import Image

PATH = './data/'  # Change this to your actual data path

BUFFER_SIZE = 1000
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10  # Cycle consistency loss weight

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_image(image_tensor):
    image_tensor = tf.squeeze(image_tensor, axis=0)  # Remove batch dimension
    image_tensor = (image_tensor + 1) * 127.5  # Denormalize to [0, 255]
    image_tensor = tf.cast(image_tensor, tf.uint8)
    return image_tensor

def save_img(image, save_to):
    # Save the generated image
    output_image_pil = Image.fromarray(image.numpy())
    output_image_pil.save(save_to)

