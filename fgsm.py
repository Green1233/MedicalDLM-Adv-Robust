#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:51:30 2020

@author: David Rodriguez
"""

from __future__ import print_function
import os
import glob
import numpy as np
import keras
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import time
import os
import numpy as np
import sys
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10, mnist
import vis
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mlxtend.plotting import plot_decision_regions
from mlxtend.feature_extraction import PrincipalComponentAnalysis
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


num_classes = 2
test_size = num_classes * 150
width = 224
height = 224
channels = 3

def build_model():
    from keras.models import load_model
    from cleverhans.utils_tf import initialize_uninitialized_global_variables

    cwd = os.getcwd()
    print('current working directory')
    print(type(cwd))
    print(cwd)

    list_of_files = glob.glob(cwd+'/cnn_models/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('latest model')
    print(latest_file)
    pathModel = latest_file

    model = load_model(pathModel)
    #K.set_learning_phase(0)
    initialize_uninitialized_global_variables(sess)

    return model

def initialize_attack(model,sess):
    from cleverhans.attacks import MadryEtAl
    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import KerasModelWrapper

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    #fgsm = MadryEtAl(wrap, sess=sess)
    #fgsm = FastGradientMethod(model, sess=sess)

    del model

    return fgsm

def createAttack(pgd, X_test, y_target, eps):

    batch_size = 50
    
    iters = 20
    step_size = (eps*0.1)
    print('step size : {0:0.8f}'.format(step_size))
    print('iteration is : {0:0.8f}'.format(iters))
    
    print('epsilon : {0:0.8f}'.format(eps))

    print("Beginning fgsm attack")

    X_test_adv_pgd = np.zeros((X_test.shape[0], width,height,channels))

    num_batches = X_test.shape[0] // batch_size

    for i in range(num_batches):

        batch_start = batch_size * i
        batch_end = batch_size * (i + 1)

        data_batch = X_test[batch_start:batch_end].reshape(-1,width,height,channels)
        label_batch = y_target[batch_start:batch_end]

        fgsm_params = {'eps': eps,
                  #'eps_iter': step_size,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  #'nb_iter': iters,
                  'y_target': label_batch}

        X_test_adv_pgd[batch_start:batch_end] = pgd.generate_np(data_batch, **fgsm_params)

    if X_test.shape[0] % batch_size:
        batch_start = (num_batches * batch_size )
        batch_end = X_test.shape[0]
        data_batch = X_test[batch_start:batch_end].reshape(-1,width,height,channels)    
        label_batch = y_target[batch_start:batch_end]
       
        fgsm_params = {'eps': eps,
                  #'eps_iter': step_size,
                  'clip_min': 0.,
                  'clip_max': 1.,
                  #'nb_iter': iters,
                  'y_target': label_batch}

        X_test_adv_pgd[batch_start:batch_end] = pgd.generate_np(data_batch, **fgsm_params)

    del pgd
    return X_test_adv_pgd


def img_to_npy(rand_num):

    import glob

    data = []
    labels = []


    melanoma_train = '/path.../experiments/datasets/chest_xray/train/pneumonia'
    benign_train = '/path.../experiments/datasets/chest_xray/train/normal'

    melanoma_valid = '/path.../experiments/datasets/chest_xray/val/pneumonia'
    benign_valid = '/path.../experiments/datasets/chest_xray/val/normal'

    melanoma_test = '/path.../experiments/datasets/chest_xray/test/pneumonia'
    benign_test = '/path.../experiments/datasets/chest_xray/test/normal'

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

    #tsne = pca_tsne(combined_data)

    #x_train_tsne, x_test_tsne, y_train_tsne, y_test_tsne = train_test_split(tsne, combined_labels, test_size=0.29, random_state=rand_num)
    #x_val_tsne, x_test_tsne, y_val_tsne, y_test_tsne = train_test_split(x_test_tsne, y_test_tsne, test_size=0.5, random_state=rand_num)

    print('After data is shuffled and split')
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    print(y_test.shape)


    return x_train, x_val, x_test, y_train, y_val, y_test, combined_data, combined_labels

def get_data(rand_num, x_test_load, y_test_load):

    np.random.seed(rand_num)
    
    print(np.unique(y_test_load))

    # create a list to store a tuple containing an array of index values for each unique label (the array is stored at index 0 of the tuple)
    unique_label_idx = []
    for (lbl_idx, unique_label) in enumerate(np.unique(y_test_load)):
        unique_label_idx.append(np.where(y_test_load == unique_label))
        print(unique_label_idx[lbl_idx][0])
        print(type(unique_label_idx[lbl_idx][0]))


    # randomly select 150 items from array at index zero of tuple generated from np.where 
    # create list to store randomly selected index values for each unique label (list of arrays)
    rand_idx_list = []
    for (tuple_idx, tuple_values) in enumerate(unique_label_idx):
        rand_idx_list.append(np.random.choice(tuple_values[0], 150, replace=False))
        print('randomly selected values for index {}'.format(tuple_idx))
        print(rand_idx_list[tuple_idx])
        print('confirmation that values belong to correct label')
        print(y_test_load[rand_idx_list[tuple_idx]])

    
    test_size = num_classes * 150
    #test_size = (np.unique(y_test_load).shape[0] * 150
    # create arrays to store randomly selected data and labels
    data = np.zeros((test_size,width, height, channels))
    labels = np.zeros((test_size,1))

    
    # use an array of index values to index specific data and labels, then assign data and labels to previously created arrays 
    for (item_idx, rand_idx_values) in enumerate(rand_idx_list):
        data[(150*item_idx) : (150*(item_idx+1))] = x_test_load[rand_idx_values]
        labels[(150*item_idx) : (150*(item_idx+1))] = y_test_load[rand_idx_values]

    print('shape of randomly selected data')
    print(data.shape)

    print('shape of randomly selected labels')
    print(labels.shape)

    y_attack = np.zeros((test_size,1))

    outputs = model.predict(data)
    least_likely_label = np.argmin(outputs, axis=1)

    # get a count of all unique labels 
    #print('Number of unique predicted labels')
    count_unique_labels(least_likely_label)
    y_attack = least_likely_label.reshape((test_size,1))
    print(y_attack.shape)
    y_attack_ohe = keras.utils.to_categorical(y_attack, num_classes, dtype='float32')
    print(y_attack_ohe.shape)
    labels_ohe = keras.utils.to_categorical(labels, num_classes, dtype='float32')
    print(labels_ohe.shape)
    
  
    #y_attack = least_likely_label.reshape((test_size,1))
    #y_attack = np.asarray(y_attack,dtype="float32")

    #y_test = labels.reshape((test_size,1))
    #y_test = np.asarray(y_test,dtype="float32")

    return data, labels, labels_ohe, y_attack, y_attack_ohe


def eval_attack(model, eps, eps_num, pgd, x_test, y_test, y_attack):


    x_test_adv = createAttack(pgd, x_test, y_attack, eps)
    print('adversarial test data shape: {}'.format(x_test_adv.shape))

    x_test_adv = x_test_adv.reshape((-1, width,height,channels))

    print('------------Test accuracy of epsilon {}/255'.format(eps_num))
    
    loss, accuracy = model.evaluate(x_test_adv, y_test, verbose=2)

    print('Loss : {0:0.3f}'.format(loss))
    print('Accuracy : {0:0.4f}'.format(accuracy))

    loss = '{:.3f}'.format(loss)
    accuracy = accuracy * 100
    accuracy = '{:.2f}'.format(accuracy)

    del model
    
    return loss, accuracy, x_test_adv

def epsilon():
    
    # get range of epsilon values
    eps_1 = []
    eps_2 = []
    eps_3 = []
    eps_4 = []

    eps_num_1 = []
    eps_num_2 = []
    eps_num_3 = []
    eps_num_4 = []

    eps_1.append(0)
    eps_num_1.append(0)

    for eps_iter in range(1, 10):
         
        #eps_1.append(eps_iter / 255)

        #eps_num_1.append('{0:0.1f}'.format(eps_iter))

        
        # epsilon range for attack
       
        eps1 = (eps_iter*.01) / 255
        eps_1.append(eps1)
        eps2 = (eps_iter*.1) / 255
        eps_2.append(eps2)
        eps3 = (eps_iter*1.0) / 255
        eps_3.append(eps3)

        #eps4 = (30) / 255
        #eps_4.append(eps4)

        # epsilon numerator for printing
        eps_num1 = (eps_iter*.01)
        eps_num1 = '{0:0.2f}'.format(eps_num1)
        eps_num_1.append(eps_num1)

        eps_num2 = (eps_iter*.1)
        eps_num2 = '{0:0.1f}'.format(eps_num2)
        eps_num_2.append(eps_num2)

        eps_num3 = (eps_iter*1.0)
        eps_num3 = '{0:0.1f}'.format(eps_num3)
        eps_num_3.append(eps_num3)

        #eps_num4 = (30)
        #eps_num4 = '{0:0.1f}'.format(eps_num4)
        #eps_num_4.append(eps_num4)

    eps4 = (10.0) / 255
    eps_4.append(eps4)
    
    eps_num4 = (10.0)
    eps_num4 = '{0:0.1f}'.format(eps_num4)
    eps_num_4.append(eps_num4)
    
    eps_1.extend(eps_2)
    eps_1.extend(eps_3)
    eps_1.extend(eps_4)

    eps_num_1.extend(eps_num_2)
    eps_num_1.extend(eps_num_3)
    eps_num_1.extend(eps_num_4)
    

    print('epsilon values')
    print(eps_1)
    print('epsilon numerator')
    print(eps_num_1)

    return eps_1, eps_num_1

def count_unique_labels(label_set):

    # get a count of all unique labels
    (unique, counts)=np.unique(label_set, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    return

####################################################################################################
def saliency_adv(model, data, labels, y_attack, eps_num, adv_all_eps):

    from vis.utils import utils
    from vis.visualization import visualize_saliency
    import scipy.ndimage as ndimage
    import os

    adv_all_eps = np.asarray(adv_all_eps, dtype='float32')


    cwd = os.getcwd()



    adversarial_data_predictions = model.predict(adv_all_eps)
    #print('predictions on adversarial data')
    #print(adversarial_data_predictions)

    clean_data_predictions = model.predict(data)
    #print('predictions on clean data')
    #print(clean_data_predictions)

    #most_likely_label = outputs.argmax()
    most_likely_label_adv = np.argmax(adversarial_data_predictions, axis=1)

    most_likely_label_clean = np.argmax(clean_data_predictions, axis=1)

    # get index of classification layer
    #model.layers[-1].activation = keras.activations.linear
    #layer_index = utils.find_layer_idx(model, model.layers[-1])
    layer_index = utils.find_layer_idx(model, 'dense_1')

    # Swap softmax with linear
    model.layers[layer_index].activation = keras.activations.linear
    model = utils.apply_modifications(model)
    
    for i in range(len(eps_num)):
        print('epsilon values in saliency function: {}'.format(eps_num))
        adv_single_eps = adv_all_eps[i*test_size:test_size*(i+1)]
        attack_labels_single_eps = most_likely_label_adv[i*test_size:test_size*(i+1)]
        adv_pred_proba = adversarial_data_predictions[i*test_size:test_size*(i+1)]

        test_idx =[0,150]
        for idx in test_idx:
        #for idx in range(300):
            
            attn_map_clean = visualize_saliency(model,
                                   layer_index,
                                   filter_indices = most_likely_label_clean[idx],
                                   #filter_indices = None,
                                   seed_input = data[idx])
    
            gaussian_attn_map_clean = ndimage.gaussian_filter(attn_map_clean, sigma=5)
            
    
            attn_map_adv = visualize_saliency(model,
                                   layer_index,
                                   filter_indices = attack_labels_single_eps[idx],
                                   #filter_indices = None,
                                   seed_input = adv_single_eps[idx])
    
            gaussian_attn_map_adv = ndimage.gaussian_filter(attn_map_adv, sigma=5) 
    


            if (attack_labels_single_eps[idx]==0):
                predicted_class_name_adv = 'Normal'
            elif (attack_labels_single_eps[idx]==1):
                predicted_class_name_adv = 'Pneumonia'



    

            if (most_likely_label_clean[idx]==0):
                predicted_class_name_clean = 'Normal'
            elif (most_likely_label_clean[idx]==1):
                predicted_class_name_clean = 'Pneumonia'





            if (labels[idx]==0):
                true_class_name = 'Normal'
            elif (labels[idx]==1):
                true_class_name = 'Pneumonia'


                
    

            if (y_attack[idx]==0):
                target_attack_class = 'Normal'
            elif (y_attack[idx]==1):
                target_attack_class = 'Pneumonia'



    
            adversarial_highest_score = adv_pred_proba[idx].max()
            print(adv_pred_proba[idx])
            adversarial_highest_score = adversarial_highest_score * 100
            print('highest accuracy score on adversarial data for index {}'.format(idx))
            adversarial_highest_score = round(adversarial_highest_score,1)
            print(adversarial_highest_score)
    
            clean_highest_score = clean_data_predictions[idx].max()
            print(clean_data_predictions[idx])
            clean_highest_score = clean_highest_score * 100
            print('highest predicted accuracy score on clean data for index {}'.format(idx))
            clean_highest_score = round(clean_highest_score,1)
            print(clean_highest_score)
    
            if (idx == 0) | (idx == 150):
                
                fig = plt.figure(figsize=(12,12))
                fig.suptitle('True Class: {}\nEpsilon: {}'.format(true_class_name, eps_num[i]), size=25)
                gs1 = gridspec.GridSpec(2, 2)
                gs1.update(wspace=0.1, hspace=0.1)
                
                ax1 = plt.subplot(gs1[0,0])
                ax1.set_title('Clean: {}'.format(predicted_class_name_clean),size=25)
                ax1.imshow(data[idx])
                ax1.text(0.04,0.05, '{}%'.format(clean_highest_score), color="white", size=25, bbox=dict(facecolor='green', alpha=0.8), transform=ax1.transAxes, horizontalalignment='left')
                
                ax2 = plt.subplot(gs1[0,1])
                ax2.set_title('Clean Saliency',size=25)
                ax2.imshow(data[idx])
                ax2.imshow(gaussian_attn_map_clean,cmap="jet", alpha=.7)
                
                ax3 = plt.subplot(gs1[1,0])
                ax3.set_title('Adversarial: {}'.format(predicted_class_name_adv),size=25)
                ax3.imshow(adv_single_eps[idx])
                ax3.text(0.04,0.05, '{}%'.format(adversarial_highest_score), color="white", size=25, bbox=dict(facecolor='red', alpha=0.8), transform=ax3.transAxes, horizontalalignment='left')
                #axes[1,0].text(0,0, '{}%'.format(predicted_accuracy), bbox=dict(facecolor='red', alpha=0.7), fontsize=10, horizontalalignment='center', verticalalignment='center', transform = axes[1,0].transAxes)
                
                ax4 = plt.subplot(gs1[1,1])
                ax4.set_title('Adversarial Saliency',size=25)
                ax4.imshow(adv_single_eps[idx])
                ax4.imshow(gaussian_attn_map_adv,cmap="jet", alpha=.7)
                
                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')
                ax4.axis('off')
                
                
                plt.show()
                
                    
                fig.savefig(cwd+"/pgd_saliency/pdf/epsilon_{}_index_{}.pdf".format(eps_num[i], idx))
                plt.clf()


    return

def pca_tsne(combined_features):
    

    
    
    #x_train_flat = high_dim_data.reshape(-1,224*224*3)
    x_train_flat = combined_features.reshape(combined_features.shape)
    #x_train_flat = combined_features.reshape(-1, combined_features.shape[1]*combined_features.shape[2]*combined_features.shape[3])


    
    tsne = TSNE(random_state=rand_num).fit_transform(x_train_flat)

    print('shape after tsne function: {}'.format(tsne.shape))
    
    return tsne


def plot_decision_boundary(adv_all_eps, x_train_tsne, x_test_tsne, tsne, model, x_test_adv_tsne, eps_num, y_attack, tsne_clean, combined_features, labels, accuracy1, all_data_pred):


    #clf = LogisticRegression(random_state=1, solver='lbfgs')
    #clf = GaussianNB()
    #clf = DecisionTreeClassifier(random_state=0)
    #clf = clf.fit(x_train_tsne, train_pred)
    #X_combined = np.vstack((x_train_tsne, train_pred)) 
    #clf = SVC(gamma='auto')
    #clf.fit(x_train_tsne, train_pred)

    # build and train Voronoi model
    #clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf = KNeighborsClassifier(n_neighbors=11)
    all_data_pred = np.argmax(all_data_pred, axis=1)
    
    #clf = voronoi.fit(x_train_tsne, train_pred)
    clf.fit(tsne, all_data_pred)
    # check Voronoi's ability to predict original model behaviour on test set
    #voronoi_test_score = voronoi_model.score(x_test_tsne, test_pred)

    # create background grid
    
    pad = 1
    resolution = 100
    print('amount of points for x and y: {}'.format(tsne[0]))
    min_x, max_x = np.min(tsne[:, 0]) - pad, np.max(tsne[:, 0]) + pad
    print('min_x & max_x: {}, {}'.format(min_x, max_x))
    min_y, max_y = np.min(tsne[:, 1]) - pad, np.max(tsne[:, 1]) + pad
    print('min_y & max_y: {}, {}'.format(min_y, max_y))


    # will return all values within a specified range for x which creates a vectore
    # the amount of vectors comes from the amount specified for y range which makes it a matrix
    # the same is true for y with respect to x range
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, resolution), np.linspace(min_y, max_y, resolution),)
    print('meshgrid xx & yy shape: {}, {}'.format(xx.shape, yy.shape))
    ######z=xx**2 + yy**2
    
    # flatten x vector and flatten y vector then combine x and y for each index of each vector to generate a single grid data point to predict label
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    print("grid_points:", grid_points.shape)

    voronoi_grid_labels = clf.predict(grid_points)

    zz = voronoi_grid_labels.reshape(xx.shape)

    ############################################################################################
    y_pred = clf.predict(x_test_adv_tsne)
    print('eps_num: {}'.format(eps_num))
    for i in range(len(eps_num)):
        print('epsilon for current iteration: {}'.format(eps_num[i]))
        f = plt.figure()
        labels = labels.reshape(-1,)
        y_pred = y_pred.reshape(-1,)
        labels = labels.astype(np.integer)
        #plot_decision_regions(x_test_adv_tsne[i*test_size:test_size*(i+1):], labels, clf=clf,  legend='best')
        plot_decision_regions(x_test_adv_tsne[i*test_size:test_size*(i+1):], y_pred[0:test_size], clf=clf,  legend='best')
        f.savefig("db_adv_pgd/pdf/adv_db/db_pgd_eps_{}.pdf".format(eps_num[i]))

        plt.show()
        plt.clf()
    return
    

def combine_data_for_tsne(model, rand_num, combined_data, combined_labels, adv_all_eps):

    adv_all_eps = np.asarray(adv_all_eps, dtype='float32')
    print('shape of combined data before adding adversarial examples: {}'.format(combined_data.shape))

    combined_data = np.append(combined_data, adv_all_eps, axis=0)
    all_data_pred = model.predict(combined_data)
    print('shape of combined data after adding all adversarial examples: {}'.format(combined_data.shape))
    #tsne = pca_tsne(combined_data)
    
    outputs = model.layers[-2].output
    print(model.layers[-2].name)

    model2 = Model(inputs=model.inputs, outputs=outputs)
    model2.summary()

    # get feature map for second to last layer
    combined_features = model2.predict(combined_data)

    print('feature extractor output shape of combined data (train, val, test & adv_test): {}'.format(combined_features.shape))

    # get tsne projection of combined features which includes all data & adv examples 
    tsne = pca_tsne(combined_features)

    print('shape of tsne after adding adversarial examples: {}'.format(tsne.shape))

    
    x_test_adv_tsne = tsne[-adv_all_eps.shape[0]:]
    tsne_clean = tsne[:-adv_all_eps.shape[0]]
    #np.delete(tsne, tsne[-300:], axis=0)
    print('shape of tsne data after removing adversarial examples: {}'.format(tsne.shape))
    print('shape of adversarial examples after removing them from combined tsne data: {}'.format(x_test_adv_tsne.shape))
    
    x_train_tsne, x_test_tsne, y_train_tsne, y_test_tsne = train_test_split(tsne_clean, combined_labels, test_size=0.29, random_state=rand_num)
    x_val_tsne, x_test_tsne, y_val_tsne, y_test_tsne = train_test_split(x_test_tsne, y_test_tsne, test_size=0.5, random_state=rand_num)

    
    return tsne_clean, tsne, x_test_adv_tsne, x_train_tsne, y_train_tsne, x_val_tsne, x_test_tsne, y_val_tsne, y_test_tsne, combined_features,all_data_pred


####################################################################################################
cwd = os.getcwd()

data_dir = ['data_1','data_2','data_3','data_4','data_5','data_6','data_7','data_8','data_9','data_10']
data_rand = [127, 95, 14, 19, 77, 42, 33, 3, 215, 28]

for (data_idx, data) in enumerate(data_dir):
    if (data in cwd):
        rand_num = data_rand[data_idx]  

print('Random number is {}'.format(rand_num))

# command line arguement to enter random number
#rand_num=sys.argv[1]
#rand_num=int(rand_num)
#rand_num = 127
sess = tf.Session()
K.set_session(sess)
model = build_model()

# Load the data.

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train, x_val, x_test, y_train, y_val, y_test, combined_data, combined_labels = mnist10_rand(rand_num, x_train, y_train, x_test, y_test)

# Load the data.
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train, x_val, x_test, y_train, y_val, y_test, combined_data, combined_labels = cifar10_rand(rand_num, x_train, y_train, x_test, y_test)
x_train, x_val, x_test, y_train, y_val, y_test, combined_data, combined_labels = img_to_npy(rand_num)
del x_val
del y_val
# get a count of all labels
print('Number of unique test labels')
count_unique_labels(y_test)

# get epsilon values for attack
eps_1, eps_num_1 = epsilon()

# instantiate attack object instance 
pgd = initialize_attack(model,sess)

# load data and prepare data for attack
data, labels, labels_ohe, y_attack, y_attack_ohe = get_data(rand_num, x_test, y_test)


del x_test
del y_test

# Input image dimensions.
input_shape = x_train.shape[1:]
print('input shape: {}'.format(input_shape))

y_train = y_train.astype(np.integer)
y_train = y_train.reshape(y_train.shape[0],)

#train_pred = model.predict(x_train)
del x_train
del y_train

loss1 = []
accuracy1 = []
elapsed1 = []

adv_all_eps = []

for eps_index, eps in enumerate(eps_1):

    print('epsilon : {0:0.8f}'.format(eps))
    start = time.time()  # start measuring time
    eps_num = eps_num_1[eps_index] 
    loss, accuracy, x_test_adv = eval_attack(model, eps, eps_num, pgd, data, labels_ohe, y_attack_ohe)
    elapsed = time.time() - start # stop the stopwatch
    elapsed = '{:.2f}'.format(elapsed)
    print('Time elapsed: {}'.format(elapsed))

    
    loss1.append(loss)
    accuracy1.append(accuracy)
    elapsed1.append(elapsed)

    print(accuracy1)
    print(loss1)
    print(elapsed1)

    for k in x_test_adv:
        adv_all_eps.append(k)
        del k

tsne_clean, tsne, x_test_adv_tsne, x_train_tsne, y_train_tsne, x_val_tsne, x_test_tsne, y_val_tsne, y_test_tsne, combined_features, all_data_pred = combine_data_for_tsne(model, rand_num, combined_data, combined_labels, adv_all_eps)
                                                                                                          
plot_decision_boundary(adv_all_eps, x_train_tsne, x_test_tsne, tsne, model, x_test_adv_tsne, eps_num_1, y_attack, tsne_clean, combined_features, labels, accuracy1, all_data_pred)

saliency_adv(model, data, labels, y_attack, eps_num_1, adv_all_eps)
             
import pandas
cwd = os.getcwd()

df = pandas.DataFrame(data={"accuracy": accuracy1})
df.to_csv(cwd+'/accuracy_pgd.csv', sep=',',index=False)

df1 = pandas.DataFrame(data={"loss": loss1})
df1.to_csv(cwd+'/loss_pgd.csv', sep=',',index=False)

df2 = pandas.DataFrame(data={"time": elapsed1})
df2.to_csv(cwd+'/time_pgd.csv', sep=',',index=False)


