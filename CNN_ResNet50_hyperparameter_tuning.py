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
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import regularizers
import keras_tuner

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Place dataset in the same folder as this file!
path = 'dataset'
if not os.path.exists('dataset'):
    print("Error: Please place dataset in the same folder as this file and name it to 'dataset'!")
    exit()

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
# uncomment to see random sample of images
'''fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data.File_Path[i]))
    ax.set_title(data.Labels[i])
plt.tight_layout()
plt.show()'''

# display bar graph of # of images per label
# uncomment to see num images per label 
'''counts = data.Labels.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=50)'''

# plt.show()

# train test split 
train_df, test_df = train_test_split(data, test_size=0.2, random_state=2)
# print(train_df)
# print(test_df)

def gen(pre, train, test):
    train_datagen = ImageDataGenerator(preprocessing_function=pre, validation_split=0.2)
    test_datagen = ImageDataGenerator(preprocessing_function=pre)
    
    train_gen = train_datagen.flow_from_dataframe(dataframe=train, x_col='File_Path', y_col='Labels', target_size=(256,256), class_mode='categorical', batch_size=32, shuffle=True, seed=567, subset='training', fill_mode="nearest")
    valid_gen = train_datagen.flow_from_dataframe(dataframe=train, x_col='File_Path', y_col='Labels', target_size=(256,256), class_mode='categorical', batch_size=32, shuffle=False, seed=567, subset='validation', fill_mode="nearest")
    test_gen = test_datagen.flow_from_dataframe(dataframe=test, x_col='File_Path', y_col='Labels', target_size=(256,256), color_mode='rgb', class_mode='categorical', batch_size=32, verbose=1, shuffle=False)
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

def build_model(hp):
    pre_model = ResNet50(input_shape=(256,256, 3), include_top=False, weights='imagenet', pooling='avg')
    pre_model.trainable = False
    inputs = pre_model.input
    x = Dense(hp.Choice('units1', [1024, 512, 256]), activation='relu')(pre_model.output)
    x = Dropout(hp.Float('dropout1', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Choice('units2', [1024, 512, 256]), activation='relu')(x)
    x = Dropout(hp.Float('dropout2', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Choice('units3', [512, 256, 128]), activation='relu')(x)
    x = Dropout(hp.Float('dropout3', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Choice('units4', [256, 128, 64]), activation='relu')(x)
    x = Dropout(hp.Float('dropout4', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Choice('units5', [64, 32]), activation='relu')(x)

    '''x = Dense(hp.Choice('units1', [1024, 512, 256]), activation='relu')(pre_model.output) # first fully connected layer
    x = Dropout(hp.Float('dropout1', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Choice('units4', [256, 128, 64]), activation='relu')(x) # second fully connected layer
    x = Dropout(hp.Float('dropout4', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Choice('units5', [64, 32]), activation='relu')(x) # third fully connected layer'''

    outputs = Dense(11, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)



    model.compile(loss = 'categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

# set up pre trained resnet50 for transfer learning
# uses imagenet dataset
if (len(tensorflow.config.list_physical_devices('GPU')) > 0):
    print("Using gpu " + str(tensorflow.config.list_physical_devices('GPU')[0]))

ResNet_pre=preprocess_input
train_gen_ResNet, valid_gen_ResNet, test_gen_ResNet = gen(ResNet_pre,train_df,test_df)
tuner = keras_tuner.Hyperband(build_model, objective='val_accuracy', directory='hyperparameter_trials', project_name='hyperparameter_trials', overwrite=False)
callback  = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')]
tuner.search(
    train_gen_ResNet,
    validation_data=valid_gen_ResNet,
    epochs=100,
    callbacks=callback,
    verbose=1
)

hyperparameters = tuner.get_best_hyperparameters()[0]
print(hyperparameters)

