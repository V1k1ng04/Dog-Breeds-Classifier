import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 

import cv2 
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
from functools import partial 

import warnings 
warnings.filterwarnings('ignore') 
AUTO = tf.data.experimental.AUTOTUNE

from zipfile import ZipFile 
data_path = 'C:/Users/Satvik/Documents/dev/Dog Breed classification/dog-breed-identification.zip'

with ZipFile(data_path, 'r') as zip: 
	zip.extractall() 
	print('The data set has been extracted.') 

df = pd.read_csv('labels.csv') 

df['filepath'] = 'train/' + df['id'] + '.jpg'


le = LabelEncoder() 
df['breed'] = le.fit_transform(df['breed']) 



features = df['filepath'] 
target = df['breed'] 
  
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.15, random_state=10) 
  

import albumentations as A 

transforms_train = A.Compose([ 
	A.VerticalFlip(p=0.2), 
	A.HorizontalFlip(p=0.7), 
	A.CoarseDropout(p=0.5), 
	A.RandomGamma(p=0.5), 
	A.RandomBrightnessContrast(p=1) 
]) 


augments = [A.VerticalFlip(p=1), A.HorizontalFlip(p=1), 
            A.CoarseDropout(p=1), A.CLAHE(p=1)]

def aug_fn(img): 
	aug_data = transforms_train(image=img) 
	aug_img = aug_data['image'] 

	return aug_img 


@tf.function 
def process_data(img, label): 
	aug_img = tf.numpy_function(aug_fn, 
								[img], 
								Tout=tf.float32) 

	return img, label 


def decode_image(filepath, label=None): 

	img = tf.io.read_file(filepath) 
	img = tf.image.decode_jpeg(img) 
	img = tf.image.resize(img, [128, 128]) 
	img = tf.cast(img, tf.float32) / 255.0

	if label == None: 
		return img 

	return img, tf.one_hot(indices=label, 
						depth=120, 
						dtype=tf.float32) 


train_ds = ( 
	tf.data.Dataset 
	.from_tensor_slices((X_train, Y_train)) 
	.map(decode_image, num_parallel_calls=AUTO) 
	.map(partial(process_data), num_parallel_calls=AUTO) 
	.batch(32) 
	.prefetch(AUTO) 
) 

val_ds = ( 
	tf.data.Dataset 
	.from_tensor_slices((X_val, Y_val)) 
	.map(decode_image, num_parallel_calls=AUTO) 
	.batch(32) 
	.prefetch(AUTO) 
) 

from keras.applications.inception_v3 import InceptionV3 

pre_trained_model = InceptionV3( 
	input_shape=(128, 128, 3), 
	weights='imagenet', 
	include_top=False
) 

print(len(pre_trained_model.layers))

for layer in pre_trained_model.layers: 
	layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7') 
print('last layer output shape: ', last_layer.output)
last_output = last_layer.output

# Model Architecture 
x = layers.Flatten()(last_output) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.BatchNormalization()(x) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dropout(0.3)(x) 
x = layers.BatchNormalization()(x) 
output = layers.Dense(120, activation='softmax')(x) 

model = keras.Model(pre_trained_model.input, output) 

# Model Compilation 
model.compile( 
	optimizer='adam', 
	loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
	metrics=['AUC'] 
) 

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
 

class myCallback(tf.keras.callbacks.Callback): 
	def on_epoch_end(self, epoch, logs={}): 
		if logs.get('val_auc') > 0.99: 
			print('\n Validation accuracy has reached upto 90% ,so, stopping further training.') 
			self.model.stop_training = True

es = EarlyStopping(patience=3, 
				monitor='val_auc', 
				restore_best_weights=True) 

lr = ReduceLROnPlateau(monitor='val_loss', 
					patience=2, 
					factor=0.5, 
					verbose=1) 

history = model.fit(train_ds, 
					validation_data=val_ds, 
					epochs=10, 
					verbose=1, 
					callbacks=[es, lr, myCallback()]) 

