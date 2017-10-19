### Aurora
"""
Neural network architecture
Reference: https://github.com/wassname/rl-portfolio-management/blob/master/keras-ddpg.ipynb
"""

# numeric
import numpy as np
from numpy import random
import pandas as pd

import tensorflow
import keras

# reinforcement learning
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from keras.models import Model, Sequential
from keras.layers import Input, InputLayer, Dense, Activation, BatchNormalization, Dropout, regularizers
from keras.layers import Conv1D, Conv2D 
from keras.layers import Flatten, Reshape, concatenate, merge
from keras.optimizers import Adam
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1_l2


#################### CNN ####################
class Network( object ):
    """Build neural network architecture."""

    def __init__( self, number_assets, window, number_features = 1 ):
        """Constructor."""

        self.number_assets = number_assets ### number of assets (excluding cash) 
        self.window = window
        self.number_features = number_features
        ### our input is a single price (or return) series, not high, low and closed prices
        ### could expand to a broader set of features
        

    def actor( self, number_samples = 1, feature_maps = ( 2, 20, 1 ), kernel_size = 3, activation = 'relu' ):
        """CNN actor model"""

        ###### 0. input layer ######
        x1 = Input( shape = ( number_samples, self.number_assets, window, self.number_features ), name = 'return time series' )
        ### last dimension -- self.number_features -- denotes channel
        x2 = Reshape( ( self.number_assets, window, self.number_features ) )( x1 )
        
        ###### 1. conv2D layer ######
        x2 = Conv2D( filters = feature_maps[0], kernel_size = ( 1, kernel_size ), kernel_regularizer = l2(reg), activation = activation )( x2 )
        x2 = BatchNormalization()( x2 )
        ###### 2. conv2D layer ######
        x2 = Conv2D( filters = feature_maps[1], kernel_size = ( 1, self.window - kernel_size + 1 ), kernel_regularizer = l2(reg), activation = activation )( x2 )
        x2 = BatchNormalization()( x2 )
        ###### Now we have 20 ( number_assets * 1 ) feature maps

        ###### previous action
        z1 = Input( shape = ( self.number_assets, ), name = 'previous action' )

        ### x2 = Flatten()( x2 ) ? 
        xx = concatenate()( [ x2, z1 ], axis = 1 )
        
        ###### 3. conv2D layer ######
        xx = Conv2D( filters = feature_maps[2], kernel_size = ( 1, 1 ), kernel_regularizer = l2(reg), activation = activation )( xx )
        xx = BatchNormalization()( xx )
        xx = Flatten()( xx )
        
        ###### add cash bias ######
        ### keras add bias by default? 
        xx = Dense( units = self.number_assets + 1, kernel_regularizer = l2(reg) )( xx )
        ###### softmax ######
        y = Activation( 'softmax' )( xx )

        model = Model( inputs = [ x1, z1 ], outputs = y )
        print( 'model summary: \n', model.summary() )
        return model
    

