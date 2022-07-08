# In[12]:

from skimage import transform
import PIL
from natsort import natsorted
import pandas
import numpy
from pathlib import Path

#Tensorflow
import tensorflow

#Fusion Models
#Using image data generator, the classes are in the order they appear in the files, but when passed to model, are shuffled. Hence, shuffle needs to be false for the test data set
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
from roc_utils import *
from sklearn import metrics

def run_testing(test_vector, LR_model):

    test_vector = test_vector.reshape(1,-1)
    LogRes_model_prediction = LR_model.predict_proba(test_vector)
    LogRes_model_prediction_argmax = LR_model.predict(test_vector)

    return (LogRes_model_prediction, LogRes_model_prediction_argmax)
