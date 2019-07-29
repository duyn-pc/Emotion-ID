# Expression-ID
Expression ID classifies the facial expression of a face image as anger, disgust, fear, happy, sad, surprise, or neutral. 

## Installation
While in directory, type "pip install -r requirements.txt" in your terminal to install the required packages. 
To train the model, download the [FER2013 dataset](https://www.kaggle.com/deadskull7/fer2013/downloads/fer2013.zip/1) and unzip it into the data folder of this project. 

## Usage
classifier.py - Will show an image in the main project diretory and attach an expression label to it. An example named "z_test_image" has already been placed in folder. Simply run classifier.py to show the image. 

To run one's own image, delete the "z_test_image" and place any jpg, jpeg, or png image in the project directory and run classifier.py.

model.py - To try to train this model, download the dataset, then run model.py. Delete the "model.best.hdf5" in data folder to train
           model from scratch. 
           Set the "fine tune" parameter to False when training from scratch. 

## Credits
Mini-Xception Model by [Octavio Arriaga et al](https://github.com/oarriaga/face_classification). 
