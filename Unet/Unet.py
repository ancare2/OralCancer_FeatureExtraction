import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

from UnetUtils import UnetUtils
UnetUtils = UnetUtils()

class Unet():
    
    """ 
    Unet Model:
        https://arxiv.org/pdf/1505.04597
    """
    
    def __init__(self, input_shape = (572, 572, 1), filters = [32, 64, 128, 256], padding = "same"):


        self.input_shape = input_shape
        self.filters = filters
        self.padding = padding
    
    def Build_UNetwork(self):
     
        
        UnetInput = Input(self.input_shape)
        
        # contracting path (3 bloques)
        conv1, pool1 = UnetUtils.contracting_block(input_layer=UnetInput, filters=self.filters[0], padding=self.padding)
        conv2, pool2 = UnetUtils.contracting_block(input_layer=pool1, filters=self.filters[1], padding=self.padding)
        conv3, pool3 = UnetUtils.contracting_block(input_layer=pool2, filters=self.filters[2], padding=self.padding)

        # bottleneck
        bottleNeck = UnetUtils.bottleneck_block(pool3, filters=self.filters[3], padding=self.padding)

        # expansive path (3 bloques)
        upConv1 = UnetUtils.expansive_block(bottleNeck, conv3, filters=self.filters[2], padding=self.padding)
        upConv2 = UnetUtils.expansive_block(upConv1, conv2, filters=self.filters[1], padding=self.padding)
        upConv3 = UnetUtils.expansive_block(upConv2, conv1, filters=self.filters[0], padding=self.padding)


        UnetOutput = Conv2D(1, (1, 1), padding = self.padding, activation = tf.math.sigmoid)(upConv3)
        
        model = Model(UnetInput, UnetOutput, name = "UNet")
        
        return model

    def CompileAndSummarizeModel(self, model, optimizer = "adam", loss = "binary_crossentropy"):
        

        model.compile(optimizer = optimizer, loss = loss, metrics = ["acc"])
        model.summary()
        
    def plotModel(self, model, to_file = 'unet.png', show_shapes = True, dpi = 96):
        
        
        tf.keras.utils.plot_model(model, to_file = to_file, show_shapes = show_shapes, dpi = dpi)