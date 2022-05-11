import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.get_logger().setLevel('ERROR') # sneaky, sneaky

TARGET_SIZE = (150,150)
BATCH_SIZE = 16
DATA_DIR = 'data'  

def get_generator():
    data_gen = ImageDataGenerator(
            rescale=1./255,
            height_shift_range=0.3,
            width_shift_range=0.3)

    img_generator = data_gen.flow_from_directory(
            DATA_DIR, 
            target_size=(TARGET_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary') 
    return img_generator
