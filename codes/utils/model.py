from .__init__ import *

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from . import efficientnet, resnext

def Model():
    
    inputs = tf.keras.layers.Input(INPUT_SHAPE)
    backbone = efficientnet.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    x = tf.keras.layers.Dropout(0.2) (x)
    x = standart_head(x)
    x = tf.keras.layers.Dense(5, activation='softmax') (x)
    model = tf.keras.Model(inputs, x)
    
    return model

def standart_head(input_tensor):
    x = tf.keras.layers.Dense(256) (input_tensor)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Activation('relu') (x)
    x = tf.keras.layers.Dropout(0.2) (x)
    return x

def spinalNet_head(x):
    
    x = tf.keras.layers.Dense(512) (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Activation('relu') (x)
    
    def linear_block(x):
        x = tf.keras.layers.Dropout(0.1) (x)
        return tf.keras.layers.Dense(x.shape[-1], activation='relu') (x)
        
    x1 = linear_block(x[:, 0:x.shape[-1]//2])
    x2 = linear_block( tf.keras.layers.Concatenate(axis=-1)([x[:, x.shape[-1]//2:x.shape[-1]], x1]) )
    x3 = linear_block( tf.keras.layers.Concatenate(axis=-1)([x[:,0:x.shape[-1]//2], x2]) )
    x4 = linear_block( tf.keras.layers.Concatenate(axis=-1)([x[:, x.shape[-1]//2:x.shape[-1]], x3]) )
    
    x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])
    x = tf.keras.layers.Concatenate(axis=-1)([x, x3])
    x = tf.keras.layers.Concatenate(axis=-1)([x, x4])
    
    x = tf.keras.layers.Dropout(0.1) (x)
    
    return x

"""
# loss function
def bi_tempered(y_true, y_pred):
    loss = bi_tempered_logistic_loss(y_pred, y_true, 0.5, 1.2, 0.2)
    return tf.reduce_sum(loss)

class_weight = np.array(list(loader.class_weight.values()))
gamma = 1./class_weight
gamma /= gamma.min()
gamma = np.round(gamma)
def focal_loss(y_true, y_pred):
    y_true = tf.math.argmax(y_true, axis=-1)
    return SparseCategoricalFocalLoss(
        gamma=gamma,
        reduction=tf.keras.losses.Reduction.NONE,
    ) (y_true, y_pred)

def complete_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + bi_tempered(y_true, y_pred)
"""