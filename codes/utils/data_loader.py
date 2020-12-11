from .__init__ import *

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

import imgaug.augmenters as iaa
import imgaug.parameters as iap

class DataLoader(object):
    def __init__(self, split_method='balanced'):
        
        # open the default training csv file
        df = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
        df['path'] = '../input/cassava-leaf-disease-classification/train_images/' + df['image_id']
        
        # split into train and validation sets
        self.split_method = split_method
        self.df_train, self.df_val = self._split_data(df, split_method)
        
        classes_to_predict = [0, 1, 2, 3, 4]
        class_weights = class_weight.compute_class_weight('balanced', classes_to_predict, np.array(self.df_train.label))
        self.class_weights = {i : class_weights[i] for i,label in enumerate(classes_to_predict)}
        print(f'Class weights: {self.class_weights}')
        
        print(f'Total train: {len(self.df_train)}, Total val: {len(self.df_val)}')
        
    def _split_data(self, df, method):
        
        assert 'balanced' in method or method=='ratio', 'Method for splitting dataset should either be balanced or ratio!'
        
        if 'balanced' in method:
            nr_samples = int(df.label.value_counts().min() * SPLIT_RATIO)
        
            df_val = []
            for c in range(CLASSES):
                df_val.append(df[df['label']==c].sample(nr_samples))
            df_val = pd.concat(df_val)
            df_train = df[[d not in list(df_val['image_id']) for d in list(df['image_id'])]]
            
        elif method=='ratio':
            df_train, df_val = train_test_split(df, test_size=SPLIT_RATIO, random_state=SEED, shuffle=True, stratify=df['label'])
            
        if method=='balanced_epochs':
            nr_samples = int(df_train.label.value_counts().max())

            dff = pd.DataFrame(columns=df.columns)
            for c in range(CLASSES):
                df_class = df_train[df_train['label']==c]
                
                ratio = max(1, int(nr_samples/len(df_class)))
                for _ in range(ratio):
                    dff = dff.append(df_class)
            df_train = dff
            
        return df_train, df_val
        
    def data_size(self, data):
        if data=='train': return len(self.df_train)
        elif data=='val': return len(self.df_val)
    
    def print_data_distribution(self, data, normalize=True):
        if data=='train':
            df = self.df_train
        elif data=='val':
            df = self.df_val
        
        if normalize:
            print(df.label.value_counts(normalize=True) * 100)
        else:
            print(df.label.value_counts())

    def augment(self, img):
        return AUG(image=img)
        
    def norm(self, img):
        return np.float32(img-DATA_MEAN)/DATA_STD
    
    def denorm(self, img):
        return np.uint8((img*DATA_STD+DATA_MEAN))
    
    @tf.autograph.experimental.do_not_convert
    def _get_generator(self, rows, augment):
        ''' Generator of data '''
        
        def generator(index):
            row = rows[index]
            
            path  = row['path']
            label = tf.keras.utils.to_categorical(row['label'], num_classes=CLASSES)
            
            img = imread(path, resize=INPUT_SHAPE[:2])
            
            if augment:
                img = self.augment(img)
            img = self.norm(img)
                
            yield img, label
            
        return generator
            
    def flow(self, data, batch_size):
        ''' Tensorflow iterator '''
        
        # verify dataset to be used
        augment = False
        if tf.equal(data, 'train'):
            df = self.df_train
            augment = True
        elif tf.equal(data, 'val'):
            df = self.df_val
        
        # row to each path sample
        if self.split_method=='balanced_epochs' and data=='train':
            df_train1, df_train2 = train_test_split(df, test_size=0.5, random_state=SEED, shuffle=True, stratify=df['label'])
            
            rows1 = [r[1] for r in list(df_train1.iterrows())]
            indexes1 = [i for i in range(len(rows1))]
            np.random.shuffle(indexes1)
            
            rows2 = [r[1] for r in list(df_train2.iterrows())]
            indexes2 = [i for i in range(len(rows2))]
            np.random.shuffle(indexes2)
            
            rows = rows1 + rows2
            indexes = indexes1 + indexes2
            
        else:
            rows = [r[1] for r in list(df.iterrows())]
            indexes = [i for i in range(len(rows))]
            np.random.shuffle(indexes)
        
        # get one data for getting shape and dtypes
        x,y = next(self._get_generator(rows, augment)(0))
        
        # Tensorflow Dataset API options
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset
            dataset = dataset.from_tensor_slices(indexes)
            dataset = dataset.interleave(lambda index:tf.data.Dataset.from_generator(self._get_generator(rows, augment),
                            (x.dtype, y.dtype),
                            output_shapes=(x.shape, y.shape),
                            args=(index,)),
                        cycle_length=1,
                        block_length=1,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()
            dataset = dataset.apply(tf.data.experimental.ignore_errors())
        
        return dataset
                    
    def view_data(self, data='train', batch_size=4):
        x, y = next(iter(self.flow(data, batch_size)))
        x = x.numpy(); y = y.numpy()
        
        print('Batch X: ', x.shape, x.min(), x.max())
        print('Batch Y: ', y.shape, y.min(), y.max())
        
        plt.figure(figsize=(30,30))
        for i in range(batch_size):
            plt.subplot(batch_size+1, 4, i+1)
            plt.imshow(self.denorm(x[i]))
            plt.axis('off')
            plt.title(f'Class: {np.argmax(y[i])}')
    
    def evaluate(self, model, batch_size=1):
        
        generator = iter(self.flow('val', batch_size=batch_size))
        pbar = tqdm(range(self.data_size('val')//batch_size))
        y_pred, y_true = [], []
        for _ in pbar:
            x,y = next(generator)
            y_pred.extend(np.argmax(model.predict(x), axis=-1))
            y_true.extend(np.argmax(y, axis=-1))
        
        print('\nClassification report:')
        print(classification_report(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        sns.heatmap(cm, annot=True)
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
    
    def create_submission(self, model, path_to_csv, path_to_images):
        
        df = pd.read_csv(path_to_csv)
        
        df = pd.read_csv(path_to_csv)
        df['path'] = path_to_images + df['image_id']
        
        submission = {'image_id':[], 'label':[]}
        
        for i,row in df.iterrows():
            img = imread(row['path'], resize=INPUT_SHAPE[:2])
            img = self.norm(img)
            
            pred = model.predict(img[np.newaxis,...])
            pred = np.argmax(pred[0])
            
            submission['label'].append(pred)
            submission['image_id'].append(row['image_id'])
        
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv', index=False)
        print(submission)
        
# augmentation options
AUG = iaa.SomeOf((1,3), [
    iaa.SomeOf((0,2), [
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.imgcorruptlike.GaussianNoise(severity=1),
        iaa.LinearContrast((0.4, 1.6))
    ]),
    iaa.Sometimes(0.3, iaa.HistogramEqualization()),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([
        iaa.Affine(scale=(0.5, 1.2)),
        iaa.Affine(scale={"x": (0.5, 1.2), "y": (0.5, 1.2)}),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        iaa.Affine(rotate=(-45, 45))
    ]),
    iaa.Sometimes(0.3, iaa.Affine(shear=(-16, 16))),
    iaa.SomeOf((0,1), [
        iaa.ScaleX((0.5, 1.5)),
        iaa.ScaleY((0.5, 1.5))
    ]),
    iaa.Sometimes(0.1, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
    iaa.Rot90([1,2,3,4])
])