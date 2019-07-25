# -*- coding: utf-8 -*-
"""
Functions for loading and preparing the dataset.

The FER2013 dataset
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
"Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.
"""
from pathlib import Path

import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical

def load_data(classes, filepath, usage='Training', debug=False):
    """Loads the dataset and reshapes the data for training or testing.
    
    Returns X, the input data for our model, and Y, the expected 
    label for each picture. 
    """
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]  
    # Shrinks dataset for ease of testing
    if debug:
        df = df[:10]
        data = df
    else:
        # Shuffles dataset
        data = df.sample(frac=1)    
        
    # Reshapes the list of pixels into 48x48 arrays  
    pixels = list(data["pixels"])    
    X = []
    for i in range(len(pixels)):
        X.append([int(num) for num in pixels[i].split()])   
    X = np.array(X)
    X = X.reshape(X.shape[0], 48, 48, 1)
    X = X.astype("float32")
    # Rescales the images
    X /= 255
    X -= 0.5
    X *= 2.0
    
    # Creates an array of "emotion" label for each input
    Y = data.emotion.values
    Y = to_categorical(Y)
    return X, Y

def save_data(X, Y, filename=''):
    """Saves the X and Y numpy arrays from load_data to user directory.
    """
    np.save(Path('data/X_' + filename), X)
    np.save(Path('data/Y_' + filename), Y)
    