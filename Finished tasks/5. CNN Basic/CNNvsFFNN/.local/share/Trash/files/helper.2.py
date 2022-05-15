#from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.get_logger().setLevel('ERROR') # sneaky, sneaky

TARGET_SIZE = (150,150)
BATCH_SIZE = 16
DATA_DIR = 'data'  
classes = ['pavlos', 'not-pavlos']

def gen_dupes():
    # create duplicate images
    BATCHES_PER_EPOCH = 300//BATCH_SIZE
    for img_class in classes:
        img = Image.open((f'{DATA_DIR}/{img_class}.jpeg'))
        for i in range(1, BATCH_SIZE*BATCHES_PER_EPOCH//2+1):
            img.thumbnail(TARGET_SIZE, Image.ANTIALIAS)
            img.save(f'{DATA_DIR}/{img_class}/{img_class}{i:0>3}.jpeg', "JPEG")

def get_generator():            
    data_gen = ImageDataGenerator(
            rescale=1./255,
            height_shift_range=0.5,
            width_shift_range=0.5)

    img_generator = data_gen.flow_from_directory(
            DATA_DIR, 
            target_size=(TARGET_SIZE),
            batch_size=BATCH_SIZE,
            classes=classes,
            class_mode='binary') 
    return img_generator
