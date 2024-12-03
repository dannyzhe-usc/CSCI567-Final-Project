import tensorflow
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import torch
import torchfile
from torchvision import models



path = '/Users/dannyzheng/Desktop/USC/CSCI 567/CSCI567_Project_Weather_Images/weather_images_dataset'
path_imgs = list(glob.glob(path+'/**/*.jpg'))
# print(path_imgs)

labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], path_imgs))
file_path = pd.Series(path_imgs, name='File_Path').astype(str)
labels = pd.Series(labels, name='Labels')
data = pd.concat([file_path, labels], axis=1)
data = data.sample(frac=1).reset_index(drop=True)
# print(data.head())

# EDA and visualization 

# show random sample of different labels 
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data.File_Path[i]))
    ax.set_title(data.Labels[i])
plt.tight_layout()
# uncomment to see random sample of images
# plt.show()

# display bar graph of # of images per label 
counts = data.Labels.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=50)
# uncomment to see num images per label 
# plt.show()

# train test split 
train_df, test_df = train_test_split(data, test_size=0.2, random_state=2)
# print(train_df)
# print(test_df)

# augment data function
def gen(pre,train,test):
    train_datagen = ImageDataGenerator(preprocessing_function=pre, validation_split=0.2)
    test_datagen = ImageDataGenerator(preprocessing_function=pre)
    
    train_gen = train_datagen.flow_from_dataframe(dataframe=train, x_col='File_Path', y_col='Labels', target_size=(100,100), class_mode='categorical', batch_size=32, shuffle=True, seed=567, subset='training', rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    valid_gen = train_datagen.flow_from_dataframe(dataframe=train, x_col='File_Path', y_col='Labels', target_size=(100,100), class_mode='categorical', batch_size=32, shuffle=False, seed=567, subset='validation', rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    test_gen = test_datagen.flow_from_dataframe(dataframe=test, x_col='File_Path', y_col='Labels', target_size=(100,100), color_mode='rgb', class_mode='categorical', batch_size=32, verbose=0, shuffle=False)
    return train_gen, valid_gen, test_gen

# visualize model perf function
def plot(history, test_gen, train_gen, model):
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])
        
    # Predict Data Test
    pred = model.predict(test_gen )
    pred = np.argmax(pred,axis=1)
    labels = (train_gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    
    # Classification report
    cm=confusion_matrix(test_df.Labels,pred)
    clr = classification_report(test_df.Labels, pred)
    print(clr)
    # Display 6 picture of the dataset with their labels
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.File_Path.iloc[i+1]))
        ax.set_title(f"True: {test_df.Labels.iloc[i+1]}\nPredicted: {pred[i+1]}")
    plt.tight_layout()
    plt.show()
        
    return history

# pritn results function
def result_test(test,model_use):
    results = model_use.evaluate(test, verbose=0)
    
    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))
    
    return results












# set up pre trained resnet50 for transfer learning
# uses imagenet dataset
# TODO: maybe use places365 dataset
ResNet_pre=preprocess_input
train_gen_ResNet, valid_gen_ResNet, test_gen_ResNet = gen(ResNet_pre,train_df,test_df)

pre_model = ResNet50(input_shape=(100,100, 3), include_top=False, weights='imagenet', pooling='avg')
pre_model.trainable = False
inputs = pre_model.input

# TODO: decide how many fully connected layers at the end 
x = Dense(100, activation='relu')(pre_model.output) # first fully connected layer
x = Dense(64, activation='relu')(x) # second fully connected layer
x = Dense(32, activation='relu')(x) # third fully connected layer

outputs = Dense(11, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

callback  = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='auto')]
ResNet_model = model

history = ResNet_model.fit(
    train_gen_ResNet,
    validation_data=valid_gen_ResNet,
    epochs=100,
    callbacks=callback,
    verbose=0
)
history_ResNet= plot(history,test_gen_ResNet,train_gen_ResNet, ResNet_model)
result_ResNet = result_test(test_gen_ResNet,ResNet_model)

