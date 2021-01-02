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

import multiprocessing
from joblib import Parallel, delayed
NUM_CORES = multiprocessing.cpu_count()

class DataLoader(object):
    def __init__(self,
                 path_csv,
                 path_images,
                 split_method='ratio'):
        
        # open the default training csv file
        df = pd.read_csv(path_csv)
        if not (path_images is None or path_images=='none'):
            df['path'] = path_images + df['image_id']
        
        # split into train and validation sets
        self.split_method = split_method
        if split_method is None or split_method=='none':
            self.df_train = df
            self.df_val = []
        self.df_train, self.df_val = self._split_data(df, split_method)
        
        classes_to_predict = list(range(CLASSES))
        class_weights = class_weight.compute_class_weight('balanced', classes_to_predict, np.array(self.df_train.label))
        self.class_weight = {i : class_weights[i] for i,label in enumerate(classes_to_predict)}
        print(f'Class weights: {self.class_weight}')
        
        print(f'Total train: {len(self.df_train)}, Total val: {len(self.df_val)}')
        
    def _split_data(self, df, method):
        
        assert method in ['balanced', 'ratio'] or 'fold' in method, 'Invalid splitting method!'
        
        if method=='balanced':
            nr_samples = int(df.label.value_counts().min() * SPLIT_RATIO)
        
            df_val = []
            for c in range(CLASSES):
                df_val.append(df[df['label']==c].sample(nr_samples))
            df_val = pd.concat(df_val)
            df_train = df[[d not in list(df_val['image_id']) for d in list(df['image_id'])]]
            
        elif method=='ratio':
            df_train, df_val = train_test_split(df, test_size=SPLIT_RATIO, random_state=SEED, shuffle=True, stratify=df['label'])
        
        elif 'fold' in method:
            fold_nr = int(method.replace('fold',''))
            print(f'Using splitted fold {fold_nr}...')
            df_train = df[df['kfold']!=fold_nr]
            df_val   = df[df['kfold']==fold_nr]
            
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
            img = imread(path)
            if augment:
                img = self.augment(img)
            img = self.norm(img)
                
            yield img, label
            
        return generator
            
    def flow(self, data, batch_size, augment=True):
        ''' Tensorflow iterator '''
        
        # verify dataset to be used
        if tf.equal(data, 'train'):
            df = self.df_train
        elif tf.equal(data, 'val'):
            df = self.df_val
        
        # row to each path sample
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
            dataset = dataset.shuffle(3*batch_size)
            dataset = dataset.batch(2*batch_size)
            if tf.equal(data, 'train'):
                dataset = dataset.map(crop_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.map(cutmix, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.map(cutmix, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.unbatch()
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()
            dataset = dataset.apply(tf.data.experimental.ignore_errors())
        
        return dataset
                    
    def view_data(self, data='train', batch_size=4):
        x, y = next(iter(self.flow(data, batch_size)))
        x = x.numpy(); y = y.numpy()
        y = np.round(y, 2)
        
        print('Batch X: ', x.shape, x.min(), x.max())
        print('Batch Y: ', y.shape, y.min(), y.max())
        
        plt.figure(figsize=(30,30))
        for i in range(batch_size):
            plt.subplot(batch_size+1, 4, i+1)
            plt.imshow(self.denorm(x[i]))
            plt.axis('off')
            plt.title(f'Class: {y[i]}')
            
    def tta_aug(self, img, tta=1):
        images = []
        if tta>1:
            with Parallel(n_jobs=NUM_CORES, prefer="threads") as parallel:
                images = parallel(delayed(self.augment)(img) for i in range(tta))
        images.append(img)
        images = np.stack(images, axis=0)
        return self.norm(images)
    
    def evaluate(self, model, mode='categorical', tta=10):
        
        assert mode=='categorical' or 'sparse' in mode, 'Invalid mode!'
        
        generator = iter(self.flow('val', batch_size=1, augment=False))
        pbar = tqdm(range(self.data_size('val')))
        y_pred, y_true = [], []
        for _ in pbar:
            x,y = next(generator)

            x = self.denorm(x[0])
            x = self.tta_aug(x, tta)
            pred = np.mean(model.predict(x), axis=0)

            y_pred.append(np.argmax(pred))
            if 'sparse' in mode:
                y_true.extend(y)
            else:
                y_true.extend(np.argmax(y, axis=-1))
        
        print('\nClassification report:')
        print(classification_report(y_true, y_pred))
        
        cm = confusion_matrix(y_true, y_pred, normalize='pred')
        sns.heatmap(cm, annot=True)
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
    
    def create_submission(self, model, path_to_csv, path_to_images, tta=10):
        
        df = pd.read_csv(path_to_csv)
        df['path'] = path_to_images + df['image_id']
        
        submission = {'image_id':[], 'label':[]}
        
        for i,row in df.iterrows():
            img = imread(row['path'], resize=INPUT_SHAPE[:2])
            img = self.tta_aug(img, tta)
            
            pred = np.mean(model.predict(img), axis=0)
            pred = np.argmax(pred)
            
            submission['label'].append(pred)
            submission['image_id'].append(row['image_id'])
        
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv', index=False)
        print(submission)
                    
# crop image to 224x224
def crop_image(image, target):
    image = tf.image.random_crop(image, [tf.shape(image)[0],*INPUT_SHAPE[:2],3])
    return image, target
                    
# resize image to 224x224
def resize_image(image, target):
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    return image, target
                    
# augmentation options
AUG = iaa.Sequential([
    iaa.SomeOf((0,1), [
        iaa.AddToBrightness((-30, 30)),
        iaa.MultiplyBrightness((0.5, 1.5)),
        iaa.MultiplySaturation((0.5, 1.5)),
        iaa.AddToSaturation((-50, 50))
    ]),
    iaa.OneOf([
        iaa.ScaleX((1.0, 1.5)),
        iaa.ScaleY((1.0, 1.5)),
        iaa.Affine(scale={"x": (1.0, 1.2), "y": (1.0, 1.2)}),
        iaa.Affine(rotate=(-20, 20)),
        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        iaa.Affine(shear=(-16, 16))
    ]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90([1,2,3,4])
])

# cutmix augmentation
beta = 0.5
r = 0.7

def rand_bbox(size, lam):

    W = size[1] 
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(image, target):
    
    lam = np.random.beta(beta, beta)
    rand_index = tf.random.shuffle(tf.range(len(target)))
    
    target_a = target
    target_b = tf.gather(target, rand_index)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)       
    image_a = image
    image_b = tf.gather(image, rand_index)
    
    mask = np.ones(image.shape[1:])
    mask[bbx1:bbx2, bby1:bby2, :] = 0          
    image_cutmix = image_a*mask + image_b*(1.-mask)
    
    target = target_a * lam + target_b * (1. - lam)
    
    return image_cutmix, target
