import keras
import tensorflow as tf
import numpy as np


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)
    image = img
    img_array = keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    prediction = model.predict(img_array)[0]
    return np.argmax(prediction) # return position of the highest probability
    
