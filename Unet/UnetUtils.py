import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

import tensorflow as tf
from tensorflow.keras.layers import Dropout

class UnetUtils():
    def __init__(self):  # puedes cambiar el valor por defecto
        #self.dropout_rate = dropout_rate
        pass
    
    def contracting_block(self, input_layer, filters, padding, kernel_size=3):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=tf.nn.relu, 
                      padding=padding)(input_layer)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=tf.nn.relu, 
                      padding=padding)(conv)
        #conv = Dropout(self.dropout_rate)(conv)  # Dropout aquí
        pool = MaxPooling2D(pool_size=2, strides=2)(conv)
        return conv, pool

    def bottleneck_block(self, input_layer, filters, padding, kernel_size=3, strides=1):
        reg = tf.keras.regularizers.l2(1e-5)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                    strides=strides, activation=tf.nn.relu,
                    kernel_regularizer=reg)(input_layer)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                    strides=strides, activation=tf.nn.relu,
                    kernel_regularizer=reg)(conv)
        #conv = Dropout(self.dropout_rate)(conv)  # Dropout aquí
        return conv

    def expansive_block(self, input_layer, skip_conn_layer, filters, padding, kernel_size=3, strides=1):
        transConv = Conv2DTranspose(filters=filters, kernel_size=(2, 2),
                                    strides=2, padding=padding)(input_layer)
        if padding == "valid":
            cropped = self.crop_tensor(skip_conn_layer, transConv)
            concat = tf.keras.layers.Concatenate()([transConv, cropped])
        else:
            concat = tf.keras.layers.Concatenate()([transConv, skip_conn_layer])
        
        up_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                         activation=tf.nn.relu)(concat)
        #up_conv = Dropout(self.dropout_rate)(up_conv)  # Dropout aquí
        up_conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                         activation=tf.nn.relu)(up_conv)
        #up_conv = Dropout(self.dropout_rate)(up_conv)  # Dropout aquí
        return up_conv
    
    def crop_tensor(self, source_tensor, target_tensor):
        target_tensor_size = target_tensor.shape[2]
        source_tensor_size = source_tensor.shape[2]
        delta = source_tensor_size - target_tensor_size
        crop_left = delta // 2
        crop_right = delta - crop_left
        cropped_source = source_tensor[:, crop_left:source_tensor_size - crop_right,
                                    crop_left:source_tensor_size - crop_right, :]
        return cropped_source
