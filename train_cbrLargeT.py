from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10, mnist
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time
import sys 
from numpy import save
#import cv2
#import matplotlib.image import mpimg
from PIL import Image

# =============================================================================
# # command line arguement to enter gpu number
# gpu_num=sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# Training parameters
batch_size = 32  
epochs = 100
data_augmentation = True
num_classes = 2
model_type = 'five_layer_cnn'

def count_unique_labels(label_set):

    # get a count of all unique labels
    (unique, counts)=np.unique(label_set, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    return



###############################################################################

def img_to_npy(rand_num):

    import glob

    data = []
    labels = []


    melanoma_train = '/home/admin1/Desktop/shamu_utsa/experiments/datasets/chest_xray/train/pneumonia'
    benign_train = '/home/admin1/Desktop/shamu_utsa/experiments/datasets/chest_xray/train/normal'

    melanoma_valid = '/home/admin1/Desktop/shamu_utsa/experiments/datasets/chest_xray/val/pneumonia'
    benign_valid = '/home/admin1/Desktop/shamu_utsa/experiments/datasets/chest_xray/val/normal'

    melanoma_test = '/home/admin1/Desktop/shamu_utsa/experiments/datasets/chest_xray/test/pneumonia'
    benign_test = '/home/admin1/Desktop/shamu_utsa/experiments/datasets/chest_xray/test/normal'

    for melanoma_train_img in glob.glob(melanoma_train+'/*'):
        image = Image.open(melanoma_train_img)
        # summarize some details about the image
        #print('summary')
        #print(image.format)
        #print(image.mode)
        #print(image.size)
        image = image.resize((224,224))
        if (image.mode=="L"):
            image = np.asarray(image)
            image = np.stack((image,)*3, axis=-1)
            #print('image shape after resize and change to 3 channels')
            #print(image.shape)
        image = np.asarray(image)
        #print('image shape after resize')
        #print(image.shape)
        data.append(image)
        labels.append([1])

    for melanoma_valid_img in glob.glob(melanoma_valid+'/*'):
        image = Image.open(melanoma_valid_img)
        image = image.resize((224,224))
       	if (image.mode=="L"):
            image = np.asarray(image)
            image = np.stack((image,)*3, axis=-1)
        image = np.asarray(image)
        data.append(image)
        labels.append([1])

    for melanoma_test_img in glob.glob(melanoma_test+'/*'):
        image = Image.open(melanoma_test_img)
        image = image.resize((224,224))
       	if (image.mode=="L"):
            image = np.asarray(image)
            image = np.stack((image,)*3, axis=-1)
        image = np.asarray(image)
        data.append(image)
        labels.append([1])

    for benign_train_img in glob.glob(benign_train+'/*'):
        image = Image.open(benign_train_img)
        image = image.resize((224,224))
       	if (image.mode=="L"):
            image = np.asarray(image)
            image = np.stack((image,)*3, axis=-1)
        image = np.asarray(image)
        data.append(image)
        labels.append([0])

    for benign_valid_img in glob.glob(benign_valid+'/*'):
        image = Image.open(benign_valid_img)
        image = image.resize((224,224))
       	if (image.mode=="L"):
            image = np.asarray(image)
            image = np.stack((image,)*3, axis=-1)
        image = np.asarray(image)
        data.append(image)
        labels.append([0])

    for benign_test_img in glob.glob(benign_test+'/*'):
        image = Image.open(benign_test_img)
        image = image.resize((224,224))
       	if (image.mode=="L"):
            image = np.asarray(image)
            image = np.stack((image,)*3, axis=-1)
        image = np.asarray(image)
        data.append(image)
        labels.append([0])


    combined_data = np.asarray(data,dtype='float32')
    combined_labels = np.asarray(labels,dtype='float32')

    combined_data = combined_data / 255

    print('combined data and label shape')
    print(combined_data.shape)
    print(combined_labels.shape)

    # get a count of all unique labels after combining
    print('Number of unique labels after combining data')
    count_unique_labels(combined_labels)

    x_train, x_test, y_train, y_test = train_test_split(combined_data, combined_labels, test_size=0.29, random_state=rand_num)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=rand_num)

    print('After data is shuffled and split')
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, x_val, x_test, y_train, y_val, y_test
    
####################################################################################################

cwd = os.getcwd()

data_dir = ['data_1','data_2','data_3','data_4','data_5','data_6','data_7','data_8','data_9','data_10']
data_rand = [127, 95, 14, 19, 77, 42, 33, 3, 215, 28]

for (data_idx, data) in enumerate(data_dir):
    if (data in cwd):
        rand_num = data_rand[data_idx]

print('Random number is {}'.format(rand_num))

# Load data.
x_train, x_val, x_test, y_train, y_val, y_test = img_to_npy(rand_num)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

###################################################################
model = keras.Sequential()
input_shape=(224,224,3)

model.add(Conv2D(32, (7,7), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3), (2,2)))

model.add(Conv2D(64, (7,7)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3), (2,2)))

model.add(Conv2D(128, (7,7)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3), (2,2)))

model.add(Conv2D(256, (7,7)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((3,3), (2,2)))

model.add(Conv2D(512, (7,7)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_classes, activation='softmax'))

###################################################################


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# command line arguement to pass cwd
cur_dir = os.getcwd()

# Prepare model model saving directory.
save_dir = os.path.join(cur_dir, 'cnn_models')
model_name = 'cbrLargeT.{epoch:03d}.h5' 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

'''
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = '%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
'''

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=2,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              verbose=2,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_val, y_val),
                        epochs=epochs, verbose=2, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import pandas

data = {'accuracy':[scores[1]], 'loss':[scores[0]]}

df = pandas.DataFrame(data)
df.to_csv(cur_dir+'/cbrLargeT_model_eval.csv', sep=',',index=False)
