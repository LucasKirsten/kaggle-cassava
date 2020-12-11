from .__init__ import *
#from . import efficientnet
from .ops_model import cbam_block

import tensorflow as tf
import tensorflow.keras.backend as K

def Model():
    
    inputs = tf.keras.layers.Input(INPUT_SHAPE)
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    #backbone.trainable = False
    
    x = backbone(inputs)
    x = cbam_block(x, ratio=1)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Dense(256) (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Activation('relu') (x)
    x = tf.keras.layers.Dense(5, activation='softmax') (x)
    model = tf.keras.Model(inputs, x)
    
    metrics = [
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        'acc'
    ]
    
    model.compile(
        tf.keras.optimizers.Adam(lr=5e-4, decay=1e-4),
        loss = 'categorical_crossentropy',
        metrics = metrics
    )
    
    return model