# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 01:01:21 2021

@author: codyr
"""

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from ImageHelpers import decode_image, get_dataset
import tensorflow as tf
from pathlib import Path
import glob
import pandas as pd
import cv2

#Variables
pizza_path = "D:\MLDATA\Pizza\ActualPizza"
other_path = "D:\MLDATA\Pizza\OtherFood"
image_size = (256,256)
# Load dataset
pizza_images_dir = Path(pizza_path)
nonpizza_images_dir = Path(other_path)

pizza_images = pizza_images_dir.glob('*.jpeg') + glob('*.jpg') + glob('*.png')
nonpizza_images = nonpizza_images_dir.glob('*.jpeg') +glob('*.jpg') +glob('*.png')

train_data = []

for img in pizza_images:
    img = cv2.resize(img, image_size)
    train_data.append((img,1))
    train_data.append((cv2.flip(img,0),1)) #flip image for extra training data
    train_data.append((cv2.flip(img,1),1))

for img in nonpizza_images:
    img = cv2.resize(img, image_size)
    train_data.append((img,0))
    train_data.append((cv2.flip(img,0),0)) #flip image for extra training data
    train_data.append((cv2.flip(img,1),0))
