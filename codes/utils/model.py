from .__init__ import *
from .losses import *
from .ops_model import cbam_block
from .loss_bitempered import bi_tempered_logistic_loss

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

def Model():
    
    inputs = tf.keras.layers.Input(INPUT_SHAPE)
    backbone = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    x = backbone(inputs)
    #x = cbam_block(x)
    x = tf.keras.layers.GlobalAveragePooling2D() (x)
    x = tf.keras.layers.Dropout(0.2) (x)
    x = tf.keras.layers.Dense(256) (x)
    x = tf.keras.layers.BatchNormalization() (x)
    x = tf.keras.layers.Activation('relu') (x)
    x = tf.keras.layers.Dropout(0.1) (x)
    x = tf.keras.layers.Dense(5, kernel_regularizer=tf.keras.regularizers.l2(0.0001)) (x)
    model = tf.keras.Model(inputs, x)
    
    loss = lambda y_true,y_pred : bi_tempered_logistic_loss(
        y_pred,
        y_true,
        t1=0.2,
        t2=4.0,
        label_smoothing=0.2
    )
    
    #loss = tf.keras.losses.CategoricalCrossentropy(
    #        from_logits=True,
    #        label_smoothing=0.2
    #)
    
    # warmup learning rate with later decay
    lr_warm = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1e-8,
        decay_steps=2000,
        end_learning_rate=5e-4,
        power=1.0,
        cycle=False
    )
    lr_down = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=5e-4,
        decay_steps=10000,
        end_learning_rate=1e-6,
        power=3.0,
        cycle=True
    )
    lr = Scheduler(2000, lr_warm, lr_down)
    
    model.compile(
        tf.keras.optimizers.Adam(lr=3e-4, decay=1e-3),
        loss = loss,
        metrics = ['acc']
    )
    
    return model

class Scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, steps_warm, lr_warm, lr_after, *args, **kwargs):
        super(Scheduler, self).__init__()
        
        self.steps_warm = steps_warm
        self.lr_warm = lr_warm
        self.lr_after = lr_after
        
    def __call__(self, step):
        return tf.where(
            tf.math.greater(step, self.steps_warm),
            self.lr_after(step-self.steps_warm),
            self.lr_warm(step)
        )