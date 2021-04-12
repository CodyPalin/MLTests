# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 02:04:02 2021

@author: codyr
"""

import tensorflow as tf

def decode_image(filename, image_type, resize_shape, channels):
    value = tf.io.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.image.decode_png(value, channels=channels)
    elif image_type == 'jpeg':
        decoded_image = tf.image.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.image.decode_image(value, channels=channels)
    
    if resize_shape is not None and image_type in ['png', 'jpeg']:
        decoded_image = tf.image.resize(decoded_image, resize_shape)
    
    return decoded_image

def get_dataset(image_paths, image_type, resize_shape, channels):
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
    
    def _map_fn(filename):
        decode_images = decode_image(filename, image_type, resize_shape, channels=channels)
        return decode_images
    
    map_dataset = dataset.map(_map_fn) # we use the map method: allow to apply the function _map_fn to all the 
    # elements of dataset 
    return map_dataset