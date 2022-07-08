# In[12]:
import warnings
warnings.filterwarnings("ignore")
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

def validate_test(test_path, output_path, devices):
    
    prediction_list_test = {}
    labels_test = []
    label_map = {"Amyloidosis": 0, "HCM": 1, "HTN": 2}

    test_data = test_path
    model_path = output_path

    #Functions_______________________________________________________________________________________________________________________________________________________________
    def image_load(filename):

        np_image = PIL.Image.open(filename)
        np_image = numpy.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (299, 299, 3))
        np_image = numpy.expand_dims(np_image, axis=0)
        return np_image

    def vstack_test(key, view):

        if len(prediction_list_test[key][view]) == 0:

            return 0

        return 1

    def check_empty_test(key):

        count = 0

        for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

            count += len(prediction_list_test[key][view])

        return count

    def averaging(array):

        col_sum = array.mean(axis=0)
        return col_sum.reshape(1, -1)

    #Models_______________________________________________________________________________________________________________________________________________________________
    print("Loading Models...")
    
    mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

    with mirrored_strategy.scope():

        AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/AP2_classifier.h5")
        AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/AP3_classifier.h5")
        AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/AP4_classifier.h5")
        PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_V_classifier.h5")
        PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_M_classifier.h5")
        PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/PLAX_classifier.h5")
        
        model_list = [AP2_model, AP3_model, AP4_model, PSAX_V_model, PSAX_M_model, PLAX_model]

        with open(str(model_path) + "/late_fusion_regressor.sav", 'rb') as f:
            
            LR_model = pickle.load(open(str(model_path) + "/late_fusion_regressor.sav", 'rb'))

    #Validation_______________________________________________________________________________________________________________________________________________________________
    
    for disease in ["Amyloidosis", "HCM", "HTN"]:
        
        for study in Path(test_data + "/{0}".format(disease)).iterdir():

            study_name = str(study).split('/')[-1]
            key = "{0}_{1}".format(disease, study_name)
            labels_test.append(label_map[disease])

            prediction_list_test[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

            for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

                if Path(test_data + "//{0}/{1}/{2}/".format(disease, study_name, view)).is_dir(): #only populate available views

                    files = []

                    for file in Path(test_data + "//{0}/{1}/{2}/".format(disease, study_name, view)).glob('**/*.png'):
                        
                        files.append(str(file).split('/')[-1])

                    files = natsorted(files)
                    
                    for file in files:

                        prediction_list_test[key][view].append(image_load(test_data + "/{0}/{1}/{2}/{3}".format(disease, study_name, view, file)))

    prediction_output_test = {}
    test_studies_count = 0

    for key in list(prediction_list_test):

        test_studies_count += 1

        if check_empty_test(key) == 0:

            test_studies_count -= 1
        
        else:

            prediction_output_test[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

            for view, model in zip(["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"], model_list):
                
                if vstack_test(key, view) != 0:

                    prediction = model.predict(numpy.vstack(prediction_list_test[key][view]))
                    prediction_output_test[key][view].append(prediction)

    #Fusion_Input_______________________________________________________________________________________________________________________________________________________________
    prediction_dataframes_test = {}

    for key in prediction_output_test:

        df_list_test = []

        for view in prediction_output_test[key]:

            result = [[] for _ in range(6)]

            if len(prediction_output_test[key][view]) != 0:
                
                disease,study_name = key.split('_')[0], key.split('_')[1]
                
                df = pandas.DataFrame(numpy.array(averaging(prediction_output_test[key][view][0])), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])

            else:

                df = pandas.DataFrame(numpy.array([[-1,-1,-1]]), columns = ['Amyloidosis_{0}'.format(view), 'HCM_{0}'.format(view), 'HTN_{0}'.format(view)])

            df_list_test.append(df)

        result = pandas.concat(df_list_test, axis=1)
        prediction_dataframes_test[key] = result

    fusion_input_test = {}

    for key in prediction_dataframes_test:
        
        fusion_input_test[key] = prediction_dataframes_test[key].to_numpy()

    labels_test = None
    test_data_set = None

    for key in fusion_input_test:

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
    
    #Fusion_Model_______________________________________________________________________________________________________________________________________________________________
    print("Printing Results...")
    
    print("Overall Test Set Accuracy Score: ", LR_model.score(test_data_set, labels_test))

    print('Test Set Confusion Matrix:')
    LogRes_model_prediction = LR_model.predict_proba(test_data_set)
    LogRes_model_prediction_argmax = LR_model.predict(test_data_set)

    conf_matrix = confusion_matrix(labels_test, LogRes_model_prediction_argmax)
    print("\t", "AMY", "HCM", "HTN")
    print("AMY: ", conf_matrix[0])
    print("HCM: ", conf_matrix[1])
    print("HTN: ", conf_matrix[2])

    LogRes_accuracy = metrics.accuracy_score(labels_test, LogRes_model_prediction_argmax)
    LogRes_accuracy_f1 = metrics.f1_score(labels_test, LogRes_model_prediction_argmax, average='weighted')
    print('Test Set Accuracy Score: ', "%.3f" % (LogRes_accuracy*100), "%")
    print('Test Set F1 Score: ', "%.3f" % (LogRes_accuracy_f1*100), "%")
    print("Classification Report")
    print(classification_report(labels_test, LogRes_model_prediction_argmax))

    return "Finished Test Data Evaluation."
