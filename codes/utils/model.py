from .__init__ import *
from .losses import *
from .ops_model import cbam_block

import tensorflow as tf
import tensorflow.keras.backend as K

def Model():
    
    inputs = tf.keras.layers.Input(INPUT_SHAPE)
    backbone = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    #backbone.trainable = False
    
    x = backbone(inputs)
    x = cbam_block(x)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    x = tf.keras.layers.Dropout(0.2) (x)
    x = tf.keras.layers.Dense(512, activation='relu') (x)
    x = tf.keras.layers.Dense(512, activation='relu') (x)
    x = tf.keras.layers.Dense(5, kernel_regularizer=tf.keras.regularizers.l2(0.0001)) (x)
    model = tf.keras.Model(inputs, x)
    
    model.compile(
        tf.keras.optimizers.Adam(lr=1e-4),
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.2
        ),
        metrics = ['acc']
    )
    
    return model