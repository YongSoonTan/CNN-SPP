# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:53:51 2018

@author: Home
"""
#import Skew
from skimage.util import random_noise
from skimage.transform import rotate
import numpy as np
from skimage import exposure
import math
import random
from PIL import Image

def imgs_gc(dataset, labels, gain=1):
    shape = dataset.shape
    np.random.seed(0)
    gamma =  np.random.uniform(0.1, 0.5, shape[2])
    imgs = []
    for i in range(shape[2]):
        image_with_gc =  exposure.adjust_gamma(dataset[:,:,i], gamma=gamma[i], gain=gain)
        imgs.append(image_with_gc)
    return np.asarray(imgs), labels

def imgs_sc(dataset, labels, gain=5):  
    shape = dataset.shape
    np.random.seed(0)
    cutoff =  np.random.uniform(0.3, 0.8, shape[2])
    imgs = []
    for i in range(shape[2]):
        image_with_sc =  exposure.adjust_sigmoid(dataset[:,:,i], cutoff=cutoff[i], gain=gain)
        imgs.append(image_with_sc)
    return np.asarray(imgs), labels
    
    
def imgs_rotate(dataset, labels):
    shape = dataset.shape
    np.random.seed(0)
    degree = np.random.randint(20, 60, int( math.ceil(shape[2]/2) ) )
    np.random.seed(1)
    degree = np.append( degree, np.random.randint(-60, -20, int( math.floor(shape[2]/2)) ) )
    random.seed(0)
    random.shuffle(degree)
    imgs = []
    for i in range(shape[2]):
        image_with_rotation = rotate(dataset[:,:,i], degree[i])
        imgs.append(image_with_rotation)
    return np.asarray(imgs), labels
    
def imgs_add_s_p(dataset, labels,  mode="s&p", amount=0.03):
    shape = dataset.shape
    imgs = []
    for i in range(shape[2]):
        image_with_random_noise = random_noise(dataset[:,:,i],mode=mode,amount=amount)
        imgs.append(image_with_random_noise)
    return np.asarray(imgs), labels

def imgs_h_flip(dataset, labels):
    shape = dataset.shape
    imgs = []
    for i in range(shape[2]):
        horizontal_flip = dataset[:,:,i][:, ::-1]
        imgs.append(horizontal_flip)
    return np.asarray(imgs), labels


#Adopted from Augmentor https://github.com/mdbloice/Augmentor
def do_shear(image):
    
    width, height = image.shape
    image = Image.fromarray(image, mode=None)
    
    """randomly choose angle
    """ 
    angle_to_shear = np.random.randint(-11, -8) 
    angle_to_shear = np.append( angle_to_shear, np.random.randint(8, 11))
    angle_to_shear = random.choice(angle_to_shear)

    """randomly choose direction
    """
    directions = ["x", "y"]
    direction = random.choice(directions)
    #print(direction)
    # We use the angle phi in radians later
    phi = math.tan(math.radians(angle_to_shear))

    if direction == "x":
        # Here we need the unknown b, where a is
        # the height of the image and phi is the
        # angle we want to shear (our knowns):
        # b = tan(phi) * a
        shift_in_pixels = phi * height

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)
        # For negative tilts, we reverse phi and set offset to 0
        # Also matrix offset differs from pixel shift for neg
        # but not for pos so we will copy this value in case
        # we need to change it
        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1
        # Note: PIL expects the inverse scale, so 1/scale_factor for example.
        transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)
        image = image.transform((int(round(width + shift_in_pixels)), height),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

        image = image.crop((abs(shift_in_pixels), 0, width, height))
        image = np.array(image.resize((width, height), resample=Image.BICUBIC))    
        return image

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0,
                            phi, 1, -matrix_offset)

        image = image.transform((width, int(round(height + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

        image = image.crop((0, abs(shift_in_pixels), width, height))
        image = np.array(image.resize((width, height), resample=Image.BICUBIC))
        return image

def imgs_shear(dataset, labels, translation=(4,1)):
    shape = dataset.shape
    imgs= []
    for i in range(shape[2]):
        Shear_image = do_shear(dataset[:,:,i])
        imgs.append(Shear_image)
    return np.asarray(imgs), labels

def do_skew(image, skew_amount=0, skew_direction=0, skew_type="RANDOM"):
       
    w, h = image.shape
    x1 = 0
    x2 = h
    y1 = 0
    y2 = w
    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
    image = Image.fromarray(image, mode=None)
    skew = skew_type

    # We have two choices now: we tilt in one of four directions
    # or we skew a corner.

    if skew == "TILT_LEFT_RIGHT" :

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left

    if skew == "CORNER":

        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    # To calculate the coefficients required by PIL for the perspective skew,
    # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
    matrix = []

    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)

    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    image = image.transform(image.size, Image.PERSPECTIVE, perspective_skew_coefficients_matrix,
                    resample=Image.BICUBIC)
    image = np.array(image)
    return image
	
def imgs_skew(dataset, labels,skew_type):
    shape = dataset.shape
    imgs= []
    
    if skew_type == "CORNER":
        max_skew_amount = 10
        np.random.seed(0)
        skew_amount =  np.random.randint(7, max_skew_amount+1, shape[2])
        np.random.seed(0)
        skew_direction =  np.random.randint(0, 8, shape[2])
    elif skew_type == "TILT_LEFT_RIGHT":
        max_skew_amount = 15
        np.random.seed(0)
        skew_amount = np.random.randint(5, max_skew_amount+1, shape[2])
        np.random.seed(0)
        skew_direction =  np.random.randint(0, 2, shape[2])
            
    for i in range(shape[2]):
        Skew_image = do_skew(dataset[:,:,i], skew_amount=skew_amount[i], skew_direction=skew_direction[i], skew_type=skew_type) #TILT_LEFT_RIGHT, CORNER
        imgs.append(Skew_image)
    return np.asarray(imgs), labels