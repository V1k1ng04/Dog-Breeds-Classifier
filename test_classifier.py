import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from zipfile import ZipFile 
import cv2 
import joblib
import albumentations as A 
from functools import partial 
import warnings 

warnings.filterwarnings('ignore') 
AUTO = tf.data.experimental.AUTOTUNE

# Data extraction
data_path = 'C:/Users/Satvik/Documents/dev/Dog Breed classification/dog-breed-identification.zip'
with ZipFile(data_path, 'r') as zip: 
    zip.extractall() 
    print('The data set has been extracted.') 

# Data preparation
df = pd.read_csv('labels.csv') 
df['filepath'] = 'train/' + df['id'] + '.jpg'

le = LabelEncoder() 
df['breed'] = le.fit_transform(df['breed']) 
joblib.dump(le, 'label_encoder.pkl')

features = df['filepath'] 
target = df['breed'] 

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.15, random_state=10) 

transforms_train = A.Compose([ 
    A.VerticalFlip(p=0.2), 
    A.HorizontalFlip(p=0.7), 
    A.CoarseDropout(p=0.5), 
    A.RandomGamma(p=0.5), 
    A.RandomBrightnessContrast(p=1) 
]) 

def aug_fn(img): 
    aug_data = transforms_train(image=img) 
    aug_img = aug_data['image'] 
    return aug_img 

@tf.function 
def process_data(img, label): 
    aug_img = tf.numpy_function(aug_fn, [img], Tout=tf.float32) 
    return aug_img, label 

def decode_image(filepath, label=None): 
    img = tf.io.read_file(filepath) 
    img = tf.image.decode_jpeg(img) 
    img = tf.image.resize(img, [128, 128]) 
    img = tf.cast(img, tf.float32) / 255.0
    if label is None: 
        return img 
    return img, tf.one_hot(indices=label, depth=120, dtype=tf.float32) 

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

# Model creation
from keras.applications.inception_v3 import InceptionV3 
pre_trained_model = InceptionV3(input_shape=(128, 128, 3), weights='imagenet', include_top=False) 
for layer in pre_trained_model.layers: 
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7') 
last_output = last_layer.output

x = layers.Flatten()(last_output) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.BatchNormalization()(x) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dropout(0.3)(x) 
x = layers.BatchNormalization()(x) 
output = layers.Dense(120, activation='softmax')(x) 

model = keras.Model(pre_trained_model.input, output) 

# Model compilation
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['AUC']) 

# Callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if logs.get('val_auc') > 0.99: 
            print('\n Validation accuracy has reached up to 90%, so stopping further training.') 
            self.model.stop_training = True

es = EarlyStopping(patience=3, monitor='val_auc', restore_best_weights=True) 
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1) 

# Model training
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1, callbacks=[es, lr, myCallback()]) 

# Save model
model.save('dog_breed_classifier_model.h5')

# Prediction function
def preprocess_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_breed(model, filepath):
    img = preprocess_image(filepath)
    predictions = model.predict(img)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
    return predicted_class

def get_breed_name(predicted_class, label_encoder):
    return label_encoder.inverse_transform([predicted_class])[0]

# Load the saved model and label encoder
model = keras.models.load_model('dog_breed_classifier_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

def predict_dog_breed(model, image_path, label_encoder):
    predicted_class = predict_breed(model, image_path)
    breed_name = get_breed_name(predicted_class, label_encoder)
    return breed_name

# Example usage
image_path = 'C:/Users/Satvik/Pictures/Saved Pictures/dogu.jpg'  # Replace with the path to your image
predicted_breed = predict_dog_breed(model, image_path, label_encoder)
print(f'The predicted breed is: {predicted_breed}')
