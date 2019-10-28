#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:32:19 2019

@author: anthrod
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import random


class DQN:
    def __init__(self, input_shape, num_outputs, learning_rate=1e-4):
        self.model = models.Sequential(
            layers=[
                layers.Conv2D(32, (8,8), activation='relu', strides=(4,4), input_shape=(input_shape[0], input_shape[1], input_shape[2]), data_format="channels_first", use_bias=True),
                layers.Conv2D(64, (4,4), activation='relu', strides=(2,2), use_bias=True),
                layers.Conv2D(64, (3,3), activation='relu', strides=(1,1), use_bias=True),
                layers.Flatten(),
                layers.Dense(512, activation='relu', use_bias=True),
                layers.Dense(num_outputs, use_bias=True)
            ]        
        )
        self.optimizer = optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
        #self.model.compile(self.optimizer)
        #print(self.model.summary())

        
if __name__ == "__main__":
    input_shape = (84, 84, 4)
    output_shape = 3
    net = DQN(input_shape, output_shape)

    import wrappers
    env = wrappers.make_env('PongNoFrameskip-v4')
    env.reset()
    state = env.step(random.choice([0,1,2]))
    print(net.forward(np.asarray(state[0]).reshape(1,84,84,4)))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    