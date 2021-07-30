# loading the necessary libraries
import datetime, os
import tensorflow as tf
%load_ext tensorboard
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import keras
import seaborn as sns
import cv2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
import seaborn as sns
import glob

main_path = "../input/chest-xray-pneumonia/chest_xray/"

# accessing the train, test, and validation data from the dataset folders
train_path = os.path.join(main_path,"train")
test_path=os.path.join(main_path,"test")
val_path=os.path.join(main_path,"val")

# within the train, test, and validation folders, access the normal versus pneumonia images
pneumonia_train_images = glob.glob(train_path+"/PNEUMONIA/*.jpeg")
normal_train_images = glob.glob(train_path+"/NORMAL/*.jpeg")

pneumonia_val_images = glob.glob(val_path+"/PNEUMONIA/*.jpeg")
normal_val_images = glob.glob(val_path+"/NORMAL/*.jpeg")

pneumonia_test_images = glob.glob(test_path+"/PNEUMONIA/*.jpeg")
normal_test_images = glob.glob(test_path+"/NORMAL/*.jpeg")

# creating a data frame that stores the correct value with each image
# assigns 0 if no pneumonia
# assigns 1 if pneumonia
data = pd.DataFrame(np.concatenate([[0]*len(normal_train_images) , [1] *  len(pneumonia_train_images)]),columns=["class"])

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10), subplot_kw={'xticks':[], 'yticks':[]})

# visualizing what the normal training images look like using matplotlib
for i, ax in enumerate(axes.flat):
    img = cv2.imread(normal_train_images[i])
    img = cv2.resize(img, (512,512))
    ax.imshow(img)
    ax.set_title("Normal")
fig.tight_layout()    
plt.show()

# visualizing what the pnuemonia training images look like using matplotlib
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    img = cv2.imread(pneumonia_train_images[i])
    img = cv2.resize(img, (512,512))
    ax.imshow(img)
    ax.set_title("Pneumonia")
fig.tight_layout()    
plt.show()

# augmenting the data to become grayscale
# this allows the computer and us to better visualize the difference
# between pnuemonia and normal chest xrays
# We can see that the features of chest are much better defined in normal images
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    img = cv2.imread(pneumonia_train_images[i])
    img = cv2.resize(img, (512,512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 80, 100)
    ax.imshow(img)
    ax.set_title("Pneumonia")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15,10), subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    img = cv2.imread(normal_train_images[i])
    img = cv2.resize(img, (512,512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 80, 100)
    ax.imshow(img)
    ax.set_title("Normal")
fig.tight_layout()    
plt.show()

val_Pneumonia = len(os.listdir(val_path+'/PNEUMONIA'))
val_Normal =len(os.listdir(val_path+'/NORMAL'))
print(f'len(val_Normal) = {val_Normal},len(val_Pneumonia)={val_Pneumonia}')

# Processing the images
# The ImageDataGenerator allows us to augment the data in real time
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   zoom_range = 0.2)
val_datagen = ImageDataGenerator(rescale = 1./255.,)
test_datagen = ImageDataGenerator(rescale = 1./255.,)


train_generator = train_datagen.flow_from_directory(train_path, batch_size=32, class_mode='binary', target_size = (220, 220))
validation_generator = val_datagen.flow_from_directory(test_path, batch_size=32, class_mode = 'binary', target_size=(220, 220))
test_generator = test_datagen.flow_from_directory(val_path,shuffle=False, batch_size=32, class_mode = 'binary', target_size=(220, 220))

# The CNN Model is based on the LeNet-5 Architecture with some modifications
model = keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(220, 220, 3)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=120, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=84, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid'))

# Format for using tensorboard
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq = 1)

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

# Fitting the model
history = model.fit(train_generator,
                    epochs=10,steps_per_epoch = 5216// 32,
                    validation_data = validation_generator,
                    validation_steps = 624 // 32, 
                    callbacks = [tb_callback])
                    
accuracy = history.history['accuracy']
val_accuracy  = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

test_loss, test_accuracy = model.evaluate(test_generator)
print(test_accuracy * 100)

%tensorboard --logdir logs

pred = model.predict(test_generator)

y_pred = []
for prob in pred:
    if prob >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
       
y_true = test_generator.classes

# Creating a confusion matrix
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_true, y_pred)
sn.heatmap(cm, annot=True,cmap="Blues")
# the 2 missed were false positives - the model predicted the image to have 
# pneumonia, but the patient was actually normal

# Printing the precision, recall, and f1 score
print(classification_report(y_true, y_pred))
# precision is true positives over selected elements
# recall is true positives over all positives
# f1 score is (2 * precision * recall) / (precision + recall)
