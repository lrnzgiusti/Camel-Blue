# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:43:10 2020

@author: logiusti
"""

import numpy as np
import re
import nrrd
import cv2
import os


nrrd_train_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\train\Tumor"
nrrd_train_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\train\NoTumor"




nrrd_valid_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\validation\Tumor"
nrrd_valid_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\validation\NoTumor"

train_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\train\Tumor"
train_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\train\NoTumor"

valid_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\validation\Tumor"
valid_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\validation\NoTumor"



for image in os.listdir(train_tumor_path):
    if re.search(r"png", image):
        img = cv2.imread(train_tumor_path + '\\' + image)
        nrrd.write(nrrd_train_tumor_path + '\\' + image + '.nrrd', img)


for image in os.listdir(train_no_tumor_path):
    if re.search(r"png", image):
        img = cv2.imread(train_no_tumor_path + '\\' + image)
        nrrd.write(nrrd_train_no_tumor_path + '\\' + image + '.nrrd', img)







for image in os.listdir(valid_tumor_path):
    if re.search(r"png", image):
        img = cv2.imread(valid_tumor_path + '\\' + image)
        nrrd.write(nrrd_valid_tumor_path + '\\' + image + '.nrrd', img)


for image in os.listdir(valid_no_tumor_path):
    if re.search(r"png", image):
        img = cv2.imread(valid_no_tumor_path + '\\' + image)
        nrrd.write(nrrd_valid_no_tumor_path + '\\' + image + '.nrrd', img)