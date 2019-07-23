# -*- coding: utf-8 -*-
"""
This model is the mini-Xception architecture from Octavio Arriaga et al. 
https://github.com/oarriaga/face_classification
"""
from pathlib import Path

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.models import Sequential
from keras.regularizers import l2
import numpy as np

import utils

# Parameters
batch_size = 32
num_epochs = 100
num_classes = 7
wait_time = 50
regularization = l2(0.01)
data_folder = Path('data/fer2013.csv')

# Loads saved numpy arrays if available. Otherwise creates it with load_data. 
try:
    X_train = np.load(Path('data/X_train.npy'))
    Y_train = np.load(Path('data/Y_train.npy'))
    X_val = np.load(Path('data/X_val.npy'))
    Y_val = np.load(Path('data/Y_val.npy'))
    X_test = np.load(Path('data/X_test.npy'))
    Y_test = np.load(Path('data/Y_test.npy'))    
    print('Numpy data loaded.')   
except:  
    print('Preparing data...')
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy',
                'Sad', 'Surprise', 'Neutral']
    
    X_train, Y_train = utils.load_data(emotions,
                                       data_folder,
                                       usage='Training')
    X_val, Y_val     = utils.load_data(emotions,
                                       data_folder,
                                       usage='PublicTest')
    X_test, Y_test   = utils.load_data(emotions,
                                           data_folder,
                                       usage='PrivateTest')
    
    utils.save_data(X_train, Y_train, 'train')
    utils.save_data(X_val, Y_val, 'val')
    utils.save_data(X_test, Y_test, 'test')
    print('Numpy data saved.')

# Preprocessing the data
datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=.1,
                             horizontal_flip=True)
    
# Mini-Xception architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), strides=(1, 1), use_bias=False,
                 kernel_regularizer=regularization, input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(8, (3, 3), strides=(1, 1), use_bias=False,
                 kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(Activation('relu'))    

# Module 1
model.add(Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(SeparableConv2D(16, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(16, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

# Module 2
model.add(Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

# Module 3
model.add(Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

# Module 4
model.add(Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                          kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(Conv2D(num_classes, (3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax', name='predictions'))

# Loading saved models, if any
try:
    model.load_weights(Path('models/model.best.hdf5'))
    print('Loading saved model...')
except:
    print('No saved models detected...')

# Optimizers    
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
log_path = Path('logs/model_log.log')
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, 
                              wait_time=int(wait_time/4), verbose=1)
model_names = 'model.best.hdf5'
model_path = 'models\\' + model_names # Windows environment
#model_path= 'models/' + model_names   # Non-Windows environment
checkpoint = ModelCheckpoint(model_path, 'val_loss', save_best_only=True)
callback_list = [checkpoint, 
                 CSVLogger(log_path, append=False),
                 EarlyStopping('val_loss', patience=wait_time),
                 reduce_lr]

# Start the training
print('Training the model...')
model.fit_generator(datagen.flow(X_train, Y_train, batch_size),
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=num_epochs, 
                    verbose=1,
                    validation_data=(X_val, Y_val), 
                    callbacks=callback_list,
                    validation_steps=len(X_val) / batch_size)




































