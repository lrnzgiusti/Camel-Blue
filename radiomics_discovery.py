#!/usr/bin/env python

from __future__ import print_function

import logging
import json
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import os
import nrrd
import numpy as np
# Get some test data


# Download the test case to temporary files and return it's location. If already downloaded, it is not downloaded again,
# but it's location is still returned.


nrrd_train_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\train\Tumor"
nrrd_train_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\train\NoTumor"


nrrd_valid_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\validation\Tumor"
nrrd_valid_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_format\validation\NoTumor"





radomics_train_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_features\train\Tumor"
radomics_train_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_features\train\NoTumor"


radomics_valid_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_features\validation\Tumor"
radomics_valid_no_tumor_path = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\data\cropped\radiomics_features\validation\NoTumor"



# Regulate verbosity with radiomics.verbosity (default verbosity level = WARNING)
# radiomics.setVerbosity(logging.INFO)

# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(additionalInfo=True)
extractor.enableAllFeatures()
extractor.enableAllImageTypes()

mask = r"C:\Users\logiusti\Lorenzo\PyWorkspace\Camel-Blue\mask.nrrd"

for imageName in os.listdir(nrrd_train_tumor_path):
    img = nrrd_train_tumor_path + "\\" + imageName
    mask_shape = tuple(nrrd.read(img)[1]['sizes'])
    mask = np.ones(mask_shape)
    nrrd.write("mask.nrrd", mask)
    print("Calculating features")
    featureVector = extractor.execute(img, "mask.nrrd")
    another_dict = dict()
    for k,v in featureVector.items():
        try:
            another_dict[k] = v.tolist()
        except:
            another_dict[k] = v
    with open(radomics_train_tumor_path + "\\" + imageName + ".json", 'w') as f:
        json.dump(another_dict, f, indent=2)





for imageName in os.listdir(nrrd_train_no_tumor_path):
    img = nrrd_train_no_tumor_path + "\\" + imageName
    mask_shape = tuple(nrrd.read(img)[1]['sizes'])
    mask = np.ones(mask_shape)
    nrrd.write("mask.nrrd", mask)
    print("Calculating features")
    featureVector = extractor.execute(img,  "mask.nrrd")
    another_dict = dict()
    for k,v in featureVector.items():
        try:
            another_dict[k] = v.tolist()
        except:
            another_dict[k] = v
    with open(radomics_train_no_tumor_path + "\\" + imageName + ".json", 'w') as f:
        json.dump(another_dict, f, indent=2)







for imageName in os.listdir(nrrd_valid_tumor_path):
    img = nrrd_valid_tumor_path + "\\" + imageName
    mask_shape = tuple(nrrd.read(img)[1]['sizes'])
    mask = np.ones(mask_shape)
    nrrd.write("mask.nrrd", mask)
    print("Calculating features")
    featureVector = extractor.execute(img,  "mask.nrrd")
    another_dict = dict()
    for k,v in featureVector.items():
        try:
            another_dict[k] = v.tolist()
        except:
            another_dict[k] = v
    with open(radomics_valid_tumor_path + "\\" + imageName + ".json", 'w') as f:
        json.dump(another_dict, f, indent=2)







for imageName in os.listdir(nrrd_valid_no_tumor_path):
    img = nrrd_valid_no_tumor_path + "\\" + imageName
    mask_shape = tuple(nrrd.read(img)[1]['sizes'])
    mask = np.ones(mask_shape)
    nrrd.write("mask.nrrd", mask)
    print("Calculating features")
    featureVector = extractor.execute(img,  "mask.nrrd")
    another_dict = dict()
    for k,v in featureVector.items():
        try:
            another_dict[k] = v.tolist()
        except:
            another_dict[k] = v
    with open(radomics_valid_no_tumor_path + "\\" + imageName + ".json", 'w') as f:
        json.dump(another_dict, f, indent=2)
