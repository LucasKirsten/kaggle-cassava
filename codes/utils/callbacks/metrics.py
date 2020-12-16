import types
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from sklearn.metrics import confusion_matrix, classification_report

class ClassificationMetrics(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, validation_steps, path_to_save, mode):
        
        self.validation_data  = validation_data
        self.validation_steps = validation_steps
        self.path_to_save     = path_to_save
        
        assert mode in ['binary', 'categorical', 'sparse_categorical'], 'Invalid mode!'
        if mode=='binary':
            self.adj_fx = self.adj_fy = lambda x: np.round(x)
        elif mode=='categorical':
            self.adj_fx = self.adj_fy = lambda x: np.argmax(x, axis=-1)
        elif mode=='sparse_categorical':
            self.adj_fx = lambda x: np.argmax(x, axis=-1)
            self.adj_fy = lambda x: x
            
        self.best = -np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        
        generator = iter(self.validation_data)
        
        y_true, y_pred = [],[]
        print(f'\nValidation {epoch+1}')
        pbar = Progbar(self.validation_steps)
        for it in range(self.validation_steps):
            x,y = next(generator)
            
            y_pred.extend(self.adj_fx(self.model.predict(x)))
            y_true.extend(self.adj_fy(y))
            
            pbar.update(current=it)
        
        print('\nClassification report:')
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print('\nConfusion matrix:')
        print(confusion_matrix(y_true, y_pred, normalize='pred'))
        
        # verify if f1 score improved
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['precision']
        
        if macro_f1>self.best:
            print(f'\nEpoch {epoch+1}: precision improved from {self.best:.3f} to {macro_f1:.3f}\n')
            
            self.best = macro_f1
            self.model.save(self.path_to_save)
