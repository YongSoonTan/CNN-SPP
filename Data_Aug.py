# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:10:00 2018

@author: Home
"""

import scipy.io as sio
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import DAOps

'''
supposed training sets are already in the folder named 'Crossval' 
images in the variable Train_Images, with format Train_Images[h,w,total_number_of_images], 
where h is the height, and w is the width of the image,
choose one file name at a time for data augmentation, then run save()
'''
path = os.path.join('CrossVal', 'D3') 
Dataset = 'D3_'
#file_name = '1st_fold_train'
#file_name = '2nd_fold_train'
#file_name = '3rd_fold_train'
#file_name = '4th_fold_train'
file_name = '5th_fold_train'

Train =sio.loadmat(os.path.join(path, Dataset+file_name+'.mat'))
Train_Images = Train['trainImages']
Train_Labels = Train['trainLabels2']
print(len(Train_Images[0][2]))

def show_images(index, operation):
    if operation == "rot":
        show_aug_img(Train_Images[:,:,index], rotate_imgs[:,:,index], op="rotated")
    if operation == "s_p":
        show_aug_img(Train_Images[:,:,index], s_p_imgs[:,:,index], op="Salt and pepper")
    if operation == "h_f":
        show_aug_img(Train_Images[:,:,index], h_flip_imgs[:,:,index], op="horizontal flip")
    if operation == "shear":
        show_aug_img(Train_Images[:,:,index], shear_imgs[:,:,index], op="shear")
    if operation == "skew_corner":
        show_aug_img(Train_Images[:,:,index], skew_corner_imgs[:,:,index], op="skew corner")
    if operation == "skew_left_right":
        show_aug_img(Train_Images[:,:,index], skew_left_right_imgs[:,:,index], op="skew left or right")    
    if operation == "gc":
        show_aug_img(Train_Images[:,:,index], gc_imgs[:,:,index], op="gamma correction")    
    if operation == "sc":
        show_aug_img(Train_Images[:,:,index], sc_imgs[:,:,index], op="sigmoid correction")
    if operation == "shear_s_p":
        show_aug_img(Train_Images[:,:,index], shear_s_p_imgs[:,:,index], op="shear with salt and pepper")
    if operation == "skew_corner_s_p":
        show_aug_img(Train_Images[:,:,index], skew_corner_s_p_imgs[:,:,index], op="skew corner with salt and pepper")
    if operation == "skew_left_right_s_p":
        show_aug_img(Train_Images[:,:,index], skew_left_right_s_p_imgs[:,:,index], op="skew left or right with salt and pepper")  
 
    
def show_aug_img(before, after, op):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(before, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(after, cmap='gray')
    ax[1].set_title(op + " image")
    if op == "Rescaled":
        #ax[0].set_xlim(0, 400)
        #ax[0].set_ylim(300, 0)
        ax[0].axis('off')
        ax[1].axis('off')
    else:        
        ax[0].axis('off')
        ax[1].axis('off')
    plt.tight_layout()

def shuffle_dataset(dataset,labels):
    ind_list = [i for i in range(len(dataset[0,2]))]
    random.seed(0)
    random.shuffle(ind_list)
    train_new  = dataset[:,:,ind_list]
    labels_new = labels[:,ind_list]
    return train_new, labels_new

def gc(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_gc, gc_imgs
    dataset_augmented_gc = DAOps.imgs_gc(Train_Images, Train_Labels)
    gc_imgs = np.swapaxes(dataset_augmented_gc[0].transpose(),1,0)
    
def sc(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_sc, sc_imgs
    dataset_augmented_sc = DAOps.imgs_sc(Train_Images, Train_Labels)
    sc_imgs = np.swapaxes(dataset_augmented_sc[0].transpose(),1,0)

def rot(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_rotate, rotate_imgs
    dataset_augmented_rotate = DAOps.imgs_rotate(Train_Images, Train_Labels)
    rotate_imgs = np.swapaxes(dataset_augmented_rotate[0].transpose(),1,0)
    
def s_p(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_s_p, s_p_imgs
    dataset_augmented_s_p = DAOps.imgs_add_s_p(Train_Images, Train_Labels)
    s_p_imgs = np.swapaxes(dataset_augmented_s_p[0].transpose(),1,0)
    
def h_f(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_h_flip, h_flip_imgs
    dataset_augmented_h_flip = DAOps.imgs_h_flip(Train_Images, Train_Labels)
    h_flip_imgs = np.swapaxes(dataset_augmented_h_flip[0].transpose(),1,0)
    
def shear(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_shear, shear_imgs
    dataset_augmented_shear = DAOps.imgs_shear(Train_Images, Train_Labels)
    shear_imgs = np.swapaxes(dataset_augmented_shear[0].transpose(),1,0)
	
def skew_corner(Train_Images=Train_Images, Train_Labels=Train_Labels, skew_type="CORNER"):
    global dataset_augmented_skew_corner, skew_corner_imgs
    dataset_augmented_skew_corner = DAOps.imgs_skew(Train_Images, Train_Labels,skew_type)
    skew_corner_imgs = np.swapaxes(dataset_augmented_skew_corner[0].transpose(),1,0)
    
def skew_left_right(Train_Images=Train_Images, Train_Labels=Train_Labels, skew_type="TILT_LEFT_RIGHT"):
    global dataset_augmented_skew_left_right, skew_left_right_imgs
    dataset_augmented_skew_left_right = DAOps.imgs_skew(Train_Images, Train_Labels,skew_type)
    skew_left_right_imgs = np.swapaxes(dataset_augmented_skew_left_right[0].transpose(),1,0)

def shear_s_p(Train_Images=Train_Images, Train_Labels=Train_Labels):
    global dataset_augmented_shear_s_p, shear_s_p_imgs
    dataset_augmented_shear_s_p = DAOps.imgs_add_s_p(shear_imgs.astype(np.float64), Train_Labels)
    shear_s_p_imgs = np.swapaxes(dataset_augmented_shear_s_p[0].transpose(),1,0)
	
def skew_corner_s_p(Train_Images=Train_Images, Train_Labels=Train_Labels, skew_type="CORNER"):
    global dataset_augmented_skew_corner_s_p, skew_corner_s_p_imgs
    dataset_augmented_skew_corner_s_p = DAOps.imgs_add_s_p(skew_corner_imgs.astype(np.float64), Train_Labels)
    skew_corner_s_p_imgs = np.swapaxes(dataset_augmented_skew_corner_s_p[0].transpose(),1,0)
    
def skew_left_right_s_p(Train_Images=Train_Images, Train_Labels=Train_Labels, skew_type="TILT_LEFT_RIGHT"):
    global dataset_augmented_skew_left_right_s_p, skew_left_right_s_p_imgs
    dataset_augmented_skew_left_right_s_p = DAOps.imgs_add_s_p(skew_left_right_imgs.astype(np.float64), Train_Labels)
    skew_left_right_s_p_imgs = np.swapaxes(dataset_augmented_skew_left_right_s_p[0].transpose(),1,0)
    
	
def save(Train_Images=Train_Images, Train_Labels=Train_Labels):
    
    gc()
    sc()
    s_p()
    shear()
    skew_corner()
    skew_left_right()
    shear_s_p()
    skew_corner_s_p()
    skew_left_right_s_p()

    if 'gc_imgs' in globals():
        Train_Images = np.append(Train_Images, gc_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_gc[1], axis=1)
        #sio.savemat('gc_imgs.mat', {'gc_imgs':gc_imgs, 'gc_imgs_labels': dataset_augmented_gc[1]},do_compression=True)		
    if 'sc_imgs' in globals():
        Train_Images = np.append(Train_Images, sc_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_sc[1], axis=1)
		#sio.savemat('sc_imgs.mat', {'sc_imgs':sc_imgs, 'sc_imgs_labels': dataset_augmented_sc[1]},do_compression=True)
    if 'rotate_imgs' in globals():
        Train_Images = np.append(Train_Images, rotate_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_rotate[1], axis=1)
		#sio.savemat('rotate_imgs.mat', {'rotate_imgs':rotate_imgs, 'rotate_imgs_labels': dataset_augmented_rotate[1]},do_compression=True)
    if 's_p_imgs' in globals():   
        Train_Images = np.append(Train_Images, s_p_imgs, axis=2) 
        Train_Labels = np.append(Train_Labels, dataset_augmented_s_p[1], axis=1) 
		#sio.savemat('s_p_imgs.mat', {'s_p_imgs':s_p_imgs, 's_p_imgs_labels': dataset_augmented_s_p[1]},do_compression=True)
    if 'h_flip_imgs' in globals():
        Train_Images = np.append(Train_Images, h_flip_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_h_flip[1], axis=1)
		#sio.savemat('h_flip_imgs.mat', {'h_flip_imgs':h_flip_imgs, 'h_flip_imgs_labels': dataset_augmented_h_flip[1]},do_compression=True)		
    if 'shear_imgs' in globals():
        Train_Images = np.append(Train_Images, shear_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_shear[1], axis=1)
		#sio.savemat('shear_imgs.mat', {'shear_imgs':shear_imgs, 'shear_imgs_labels': dataset_augmented_shear[1]},do_compression=True)
    if 'skew_corner_imgs' in globals():
        Train_Images = np.append(Train_Images, skew_corner_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_skew_corner[1], axis=1)
		#sio.savemat('skew_corner_imgs.mat', {'skew_corner_imgs':skew_corner_imgs, 'skew_corner_imgs_labels': dataset_augmented_skew_corner[1]},do_compression=True)
    if 'skew_left_right_imgs' in globals():
        Train_Images = np.append(Train_Images, skew_left_right_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_skew_left_right[1], axis=1)
		#sio.savemat('skew_left_right_imgs.mat', {'skew_left_right_imgs':skew_left_right_imgs, 'skew_left_right_imgs_labels': dataset_augmented_skew_left_right[1]},do_compression=True)
        
    if 'shear_s_p_imgs' in globals():
        Train_Images = np.append(Train_Images, shear_s_p_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_shear_s_p[1], axis=1)
		#sio.savemat('shear_s_p_imgs.mat', {'shear_s_p_imgs':shear_s_p_imgs, 'shear_s_p_imgs_labels': dataset_augmented_shear_s_p[1]},do_compression=True)
    if 'skew_corner_s_p_imgs' in globals():
        Train_Images = np.append(Train_Images, skew_corner_s_p_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_skew_corner_s_p[1], axis=1)
		#sio.savemat('skew_corner_imgs.mat', {'skew_corner_s_p_imgs':skew_corner_s_p_imgs, 'skew_corner_s_p_imgs_labels': dataset_augmented_skew_corner_s_p[1]},do_compression=True)
    if 'skew_left_right_s_p_imgs' in globals():
        Train_Images = np.append(Train_Images, skew_left_right_s_p_imgs, axis=2)
        Train_Labels = np.append(Train_Labels, dataset_augmented_skew_left_right_s_p[1], axis=1)
		#sio.savemat('skew_left_right_imgs.mat', {'skew_left_right_s_p_imgs':skew_left_right_s_p_imgs, 'skew_left_right_s_p_imgs_labels': dataset_augmented_skew_left_right_s_p[1]},do_compression=True)
  
    Train_Images, Train_Labels =  shuffle_dataset(Train_Images,Train_Labels)
    
    sio.savemat('Augmented_'+Dataset+file_name+'.mat', {'trainImages':Train_Images, 'trainLabels2': Train_Labels},do_compression=True)
    
