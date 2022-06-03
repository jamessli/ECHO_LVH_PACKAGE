#!/usr/bin/env python
# coding: utf-8

# In[12]:

from skimage import transform
import PIL
from natsort import natsorted
import logging
import pandas
import numpy
import matplotlib as mpl
import matplotlib.pyplot as pyplot
from pathlib import Path

#Tensorflow
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Fusion Models
#Using image data generator, the classes are in the order they appear in the files, but when passed to model, are shuffled. Hence, shuffle needs to be false for the test data set
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report
import itertools
from roc_utils import *
import os
import argparse
import zipfile 
import logging
import requests
from tqdm import tqdm

#The EfficientnetB5 (Great for resolutions around 456 by 456, maybe switch to b7 depending on performance) B4 gives 380 by 380
from tensorflow.keras.applications import InceptionResNetV2

MODEL_TO_URL = {
	'models': 'https://drive.google.com/drive/folders/1p4-gtpfp7wlHnmlqADzVF3b-ZAJIl3H2',
}

def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={ 'id' : id }, stream=True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def download_model(name='models'):
	project_dir = os.path.dirname(os.path.abspath(__file__))

	
	os.makedirs(os.path.join(project_dir, "..", name), exist_ok=True)
	file_destination = os.path.join(project_dir, "..", 'models.zip')
	file_id = MODEL_TO_URL[name].split('id=')[-1]
	logging.info(f'Downloading {name} model (~1000MB tar.xz archive)')
	download_file_from_google_drive(file_id, file_destination)

	logging.info('Extracting model from archive (~1300MB folder)')
	with zipfile.ZipFile(file_destination, 'r') as zip_ref:
		zip_ref.extractall(path=os.path.dirname(file_destination))

	logging.info('Removing archive')
	os.remove(file_destination)
	logging.info('Done.')

def main():
	download_model()

def train_fusion(studies_path, output_path, devices):

    mirrored_strategy = tensorflow.distribute.MirroredStrategy()

    training_path_views = "{0}/{1}".format(studies_path, "study_training")
    validation_path_views = "{0}/{1}".format(studies_path, "study_validation")
    model_path = "{0}/{1}".format(studies_path, "models")

    batch_size = 32 #4Gpus, 64 batches each?
    epochs = 15
    image_size = 299
    weight_decay = .3

    training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 45, width_shift_range = .5, height_shift_range = .15, horizontal_flip = False, fill_mode = 'nearest') #Rescale images to 0 to .255

    with mirrored_strategy.scope():

        AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/GENERIC_multiclassifier.h5")
        AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/GENERIC_multiclassifier.h5")
        AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/GENERIC_multiclassifier.h5")
        PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/GENERIC_multiclassifier.h5")
        PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/GENERIC_multiclassifier.h5")
        PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/GENERIC_multiclassifier.h5")

    model_list = [AP2_model, AP3_model, AP4_model, PSAX_V_model, PSAX_M_model, PLAX_model]

    for view,model in zip(['AP2', 'AP3', 'AP4', 'PSAX_V', 'PSAX_M', 'PLAX'], model_list):

        print(view)

        training_data_gen = training_data_generator.flow_from_directory(str(training_path_views + '/{0}/'.format(view)),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
        train_len = len(training_data_gen.classes)
        print("Training data: ", training_data_gen.class_indices)

        validation_data_gen = training_data_generator.flow_from_directory(str(validation_path_views + '/{0}/'.format(view)),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
        val_len = len(validation_data_gen.classes)
        print("Validation data: ", validation_data_gen.class_indices)

        make_trainable = len(model.layers)//2

        for layer in model.layers[-make_trainable:]:

            layer.trainable = True

            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.kernel))
            
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.bias))
                
        print(model.summary())

        reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 5, factor = .5, min_lr = .000015625, verbose = 2)

        model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = .0005), loss = "categorical_crossentropy", metrics = ["accuracy"])

        history = model.fit(training_data_gen, epochs = epochs, steps_per_epoch = train_len//batch_size, validation_data=validation_data_gen, validation_steps = val_len//batch_size, verbose = 1, callbacks = [reduce_lr], use_multiprocessing = True, workers = 64)
        model.save(model_path + "/{0}_classifier.h5".format(view))

        print(history.history["accuracy"], history.history["val_accuracy"])

    def image_load(filename):

        np_image = PIL.Image.open(filename)
        np_image = numpy.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (299, 299, 3))
        np_image = numpy.expand_dims(np_image, axis=0)
        return np_image

    with mirrored_strategy.scope():

        AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/AP2_classifier.h5")
        AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/AP3_classifier.h5")
        AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/AP4_classifier.h5")
        PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_V_classifier.h5")
        PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_M_classifier.h5")
        PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/PLAX_classifier.h5")

    model_list = [AP2_model, AP3_model, AP4_model, PSAX_V_model, PSAX_M_model, PLAX_model]

    training_data = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_training/grouped_data'
    validation_data = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_validation/grouped_data'
    dataframe_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/dataframes'

    prediction_list_training = {}
    labels_training = []

    label_map = {"Amyloidosis": 0, "HCM": 1, "HTN": 2}

    for disease in ["Amyloidosis", "HCM", "HTN"]:
        
        for study in Path(training_data + "/{0}".format(disease)).iterdir():

            study_name = str(study).split('/')[-1]
            key = "{0}_{1}".format(disease, study_name)
            labels_training.append(label_map[disease])
            print(key)

            prediction_list_training[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

            for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

                files = []

                for file in Path(training_data + "//{0}/{1}/{2}/".format(disease, study_name, view)).glob('**/*.png'):
                    
                    files.append(str(file).split('/')[-1])

                files = natsorted(files)
                
                for file in files:

                    prediction_list_training[key][view].append(image_load(training_data + "/{0}/{1}/{2}/{3}".format(disease, study_name, view, file)))


    prediction_list_test = {}
    labels_test = []

    label_map = {"Amyloidosis": 0, "HCM": 1, "HTN": 2}

    for disease in ["Amyloidosis", "HCM", "HTN"]:
        
        for study in Path(validation_data + "/{0}".format(disease)).iterdir():

            study_name = str(study).split('/')[-1]
            key = "{0}_{1}".format(disease, study_name)
            labels_test.append(label_map[disease])
            print(key)

            prediction_list_test[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

            for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

                files = []

                for file in Path(validation_data + "//{0}/{1}/{2}/".format(disease, study_name, view)).glob('**/*.png'):
                    
                    files.append(str(file).split('/')[-1])

                files = natsorted(files)
                
                for file in files:

                    prediction_list_test[key][view].append(image_load(validation_data + "/{0}/{1}/{2}/{3}".format(disease, study_name, view, file)))

    def vstack_training(key, view):

        if len(prediction_list_training[key][view]) == 0:

            return 0

        return 1

    def check_empty_training(key):

        count = 0

        for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

            count += len(prediction_list_training[key][view])

        return count

    prediction_output_training = {}
    training_studies_count = 0

    for key in list(prediction_list_training):

        training_studies_count += 1

        if check_empty_training(key) == 0:

            print("Rejected: ", key)
            #prediction_list_training.pop(key)
            training_studies_count -= 1
        
        else:

            prediction_output_training[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

            print("Processing: ", key)

            for view, model in zip(["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"], model_list):
                
                if vstack_training(key, view) != 0:

                    prediction = model.predict(numpy.vstack(prediction_list_training[key][view]))
                    prediction_output_training[key][view].append(prediction)



    def vstack_test(key, view):

        if len(prediction_list_test[key][view]) == 0:

            return 0

        return 1

    def check_empty_test(key):

        count = 0

        for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

            count += len(prediction_list_test[key][view])

        return count

    prediction_output_test = {}
    test_studies_count = 0

    for key in list(prediction_list_test):

        test_studies_count += 1

        if check_empty_test(key) == 0:

            print("Rejected: ", key)
            test_studies_count -= 1
        
        else:

            prediction_output_test[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

            print("Processing: ", key)

            for view, model in zip(["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"], model_list):
                
                if vstack_test(key, view) != 0:

                    prediction = model.predict(numpy.vstack(prediction_list_test[key][view]))
                    prediction_output_test[key][view].append(prediction)

    def averaging(array):

        col_sum = array.mean(axis=0)
        return col_sum.reshape(1, -1)

    prediction_dataframes_training = {}

    for key in prediction_output_training:

        df_list = []

        for view in prediction_output_training[key]:

            if len(prediction_output_training[key][view]) != 0:
                
                disease,study_name = key.split('_')[0], key.split('_')[1]
                
                df = pandas.DataFrame(numpy.array(prediction_output_training[key][view][0]), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])
                Path(dataframe_path + "/{0}/{1}/{2}".format(disease, study_name, view)).mkdir(parents = True, exist_ok = True)
                df.to_csv(dataframe_path + "/{0}/{1}/{2}/training.csv".format(disease, study_name, view))

                df = pandas.DataFrame(numpy.array(averaging(prediction_output_training[key][view][0])), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])

            else:

                df = pandas.DataFrame(numpy.array([[-1,-1,-1]]), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])
            
            df_list.append(df)

        result = pandas.concat(df_list, axis=1)
        prediction_dataframes_training[key] = result
        print(key)

    prediction_dataframes_test = {}

    for key in prediction_output_test:

        df_list_test = []

        for view in prediction_output_test[key]:

            result = [[] for _ in range(6)]

            if len(prediction_output_test[key][view]) != 0:
                
                disease,study_name = key.split('_')[0], key.split('_')[1]
                
                df = pandas.DataFrame(numpy.array(prediction_output_test[key][view][0]), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])
                Path(dataframe_path + "/{0}/{1}/{2}".format(disease, study_name, view)).mkdir(parents = True, exist_ok = True)
                df.to_csv(dataframe_path + "/{0}/{1}/{2}/test.csv".format(disease, study_name, view))
                df = pandas.DataFrame(numpy.array(averaging(prediction_output_test[key][view][0])), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])

            else:

                df = pandas.DataFrame(numpy.array([[-1,-1,-1]]), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])

            df_list_test.append(df)

        result = pandas.concat(df_list_test, axis=1)
        prediction_dataframes_test[key] = result
        print(key)

    fusion_input_training = {}

    for key in prediction_dataframes_training:
        
        fusion_input_training[key] = prediction_dataframes_training[key].to_numpy()

    fusion_input_test = {}

    for key in prediction_dataframes_test:
        
        fusion_input_test[key] = prediction_dataframes_test[key].to_numpy()

    labels_training = None
    training_data_set = None

    for key in fusion_input_training:
        
        print(key,len(fusion_input_training[key]))

        if str(key).split('_')[0] == "Amyloidosis":

            val = 0

        elif str(key).split('_')[0] == "HCM":

            val = 1

        elif str(key).split('_')[0] == "HTN":

            val = 2

        if training_data_set is None:

            training_data_set = fusion_input_training[key]
            labels_training = numpy.full((len(fusion_input_training[key]),1), val, dtype=int)

        else:

            training_data_set = numpy.concatenate((training_data_set, fusion_input_training[key]))
            labels_training = numpy.concatenate((labels_training, numpy.full((len(fusion_input_training[key]),1), val, dtype=int)))
            
    labels_test = None
    test_data_set = None

    for key in fusion_input_test:
        
        print(key,len(fusion_input_test[key]))

        if str(key).split('_')[0] == "Amyloidosis":

            val = 0

        elif str(key).split('_')[0] == "HCM":

            val = 1

        elif str(key).split('_')[0] == "HTN":

            val = 2

        if test_data_set is None:

            test_data_set = fusion_input_test[key]
            labels_test = numpy.full((len(fusion_input_test[key]),1), val, dtype=int)

        else:

            test_data_set = numpy.concatenate((test_data_set, fusion_input_test[key]))
            labels_test = numpy.concatenate((labels_test, numpy.full((len(fusion_input_test[key]),1), val, dtype=int)))
            
    LR_model = LogisticRegression(solver = "saga", multi_class= 'ovr', C=10000, penalty='l2',max_iter = 10000)
    LR_model.fit(training_data_set, labels_training)
    score = LR_model.score(test_data_set, labels_test)
    print(score)

    from sklearn.metrics import accuracy_score
    print('Confusion Matrix')
    LogRes_model_prediction = LR_model.predict(test_data_set)
    print(accuracy_score(labels_test, LogRes_model_prediction))
    #pred_m1 = numpy.argmax(LogRes_model_prediction, axis = 1)
    print(confusion_matrix(labels_test, LogRes_model_prediction))

    LogRes_accuracy = metrics.accuracy_score(labels_test, LogRes_model_prediction)
    LogRes_accuracy_f1 = metrics.f1_score(labels_test, LogRes_model_prediction, average='weighted')
    print('Accuracy LogRes: ', "%.4f" % (LogRes_accuracy*100))
    print('F1 LogRes: ', "%.4f" % (LogRes_accuracy_f1*100))

    import pickle
    filename = 'late_fusion_regressor.sav'
    pickle.dump(LR_model, open(filename, 'wb'))
