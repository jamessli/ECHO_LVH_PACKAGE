import warnings
warnings.filterwarnings("ignore")
import pandas
from pathlib import Path
import numpy as np
import pickle
import os
#Tensorflow
import tensorflow

from sklearn.metrics import confusion_matrix, classification_report
from roc_utils import *

from skimage import transform
from PIL import Image
import PIL
from sklearn.metrics import classification_report
import numpy as numpy

#Utilizing Jason's ablation function
def generate_ablated(test_path, ablations, models, dataframes, output, devices):

    #Functions___________________________________________________________________________________________________________________________________
    def image_load(filename):

        np_image = PIL.Image.open(filename)
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (299, 299, 3))
        np_image = np.expand_dims(np_image, axis=0)

        return np_image

    def ablate_cardiac_echo(frame_center, quadrant=None):

        frame_crop = frame_center.copy()
        if quadrant != None:
            frame_center_h = int(frame_crop.shape[0] * 0.6)
            frame_center_w = int(frame_crop.shape[1] * 0.6)
            if quadrant in [1, 2, 3, 4]:  # removing a quadrant
                if quadrant == 1:
                    frame_crop[:frame_center_h, :frame_center_w] = 0
                elif quadrant == 2:
                    frame_crop[frame_center_h:, :frame_center_w] = 0
                elif quadrant == 3:
                    frame_crop[:frame_center_h, frame_center_w:] = 0
                else:
                    frame_crop[frame_center_h:, frame_center_w:] = 0
            elif quadrant in [5, 6, 7, 8]:  # keeping only the quadrant
                zeros = np.zeros(frame_crop.shape)
                if quadrant == 5:
                    zeros[:frame_center_h, :frame_center_w] = 1
                elif quadrant == 6:
                    zeros[frame_center_h:, :frame_center_w] = 1
                elif quadrant == 7:
                    zeros[:frame_center_h, frame_center_w:] = 1
                else:
                    zeros[frame_center_h:, frame_center_w:] = 1
                frame_crop = frame_crop * zeros
            else:  # this is removing or keeping the cross only
                frame_center_h_buffer = int(frame_crop.shape[0] * 0.05)
                frame_center_w_buffer = int(frame_crop.shape[1] * 0.05)
                if quadrant == 'cross_only':  # only keeping the cross
                    ones = np.ones(frame_crop.shape)
                    ones[:frame_center_h-frame_center_h_buffer, :frame_center_w-frame_center_w_buffer] = 0
                    ones[frame_center_h+frame_center_h_buffer:, :frame_center_w-frame_center_w_buffer] = 0
                    ones[:frame_center_h-frame_center_h_buffer, frame_center_w+frame_center_w_buffer:] = 0
                    ones[frame_center_h+frame_center_h_buffer:, frame_center_w+frame_center_w_buffer:] = 0
                    frame_crop = frame_crop * ones
                else:  # removing the cross only
                    zeros = np.zeros(frame_crop.shape)
                    zeros[:frame_center_h-frame_center_h_buffer, :frame_center_w-frame_center_w_buffer] = 1
                    zeros[frame_center_h+frame_center_h_buffer:, :frame_center_w-frame_center_w_buffer] = 1
                    zeros[:frame_center_h-frame_center_h_buffer, frame_center_w+frame_center_w_buffer:] = 1
                    zeros[frame_center_h+frame_center_h_buffer:, frame_center_w+frame_center_w_buffer:] = 1
                    frame_crop = frame_crop * zeros
                    
        return frame_crop

    #Variables___________________________________________________________________________________________________________________________________
    model_path = models
    image_path = test_path
    ablation_path = ablations
    dataframe_path = dataframes

    quadrants = [1, 2, 3, 4, 5, 6, 7, 8, 'cross_only', 'no_cross']
    disease_map = {'Amyloidosis':0, 'HCM':1, 'HTN': 2}

    #Create_and_Save_Ablated_images______________________________________________________________________________________________________________
    for study in Path(image_path).iterdir():

        study_name = str(study).split(os.sep)[-1]

        for disease in study.iterdir():

            disease_name = str(disease).split(os.sep)[-1]

            for view in disease.iterdir():

                path = str(view)
                view_name = path.split(os.sep)[-1]
    
                image_files = list(view.glob("*.png") )

                for quadrant in quadrants:

                    save_path = ablation_path + "/{0}/{1}/{2}/removed_{3}".format(study_name, disease_name, view_name,quadrant)
                    Path(save_path).mkdir(parents=True,exist_ok=True)

                    for img_path in image_files:

                        image = str(img_path)
                        image_name = image.split(os.sep)[-1]

                        img = Image.open(image)
                        img = img.convert('RGB')
                        image_np = ablate_cardiac_echo(np.array(img), quadrant=quadrant)
                        img = Image.fromarray(image_np.astype(np.uint8))
                        img = img.convert('RGB')
                        img.save(save_path + '/{0}.png'.format(image_name))
    
    mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

    with open(model_path + '/late_fusion_regressor.sav', 'rb') as input_file:

        LR_model = pickle.load(input_file)

    with mirrored_strategy.scope():

        AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/AP2_classifier.h5")
        AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/AP3_classifier.h5")
        AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/AP4_classifier.h5")
        PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_V_classifier.h5")
        PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_M_classifier.h5")
        PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/PLAX_classifier.h5")

        model_list = {"AP2": AP2_model, "AP3": AP3_model, "AP4":AP4_model, "PSAX_V": PSAX_V_model,"PSAX_M": PSAX_M_model,"PLAX": PLAX_model}

        template = {'Y_pred':[], 'Y_test':[]}

        ap4_ablation_dataframes = {}
        dtypes = {'Y_pred': int, 'Y_test': int}
        quadrants = set()

        for disease in Path(ablation_path).iterdir():

            label = disease_map[str(disease).rsplit(os.sep,1)[1]]

            for study in disease.iterdir():

                print(study)

                for ablation in study.joinpath('AP4').iterdir():

                    quadrant = str(ablation).rsplit(os.sep, 1)[-1]

                    if quadrant not in quadrants:

                        quadrants.add(quadrant)
                        ap4_ablation_dataframes[quadrant] = pandas.DataFrame(template)
                        ap4_ablation_dataframes[quadrant] = ap4_ablation_dataframes[quadrant].astype(dtypes)
                    
                    files = list(ablation.glob("*.png"))

                    for file_path in files:

                        prediction = AP4_model.predict(image_load(str(file_path)))
                        #prediction = np.argmax(prediction)
                        ap4_ablation_dataframes[quadrant] = pandas.concat([pandas.DataFrame([[prediction,label]], columns=['Y_pred', 'Y_test']), ap4_ablation_dataframes[quadrant]], ignore_index=True)

        for quadrant in quadrants:

            ap4_ablation_dataframes[quadrant].to_csv(dataframe_path + '/{0}_proba_predictions.csv'.format(quadrant))

        ap4_ablation_dataframes_argmax = {}
        ap4_ablation_dataframes = {}

        for quadrant in ['removed_1', 'removed_2', 'removed_3', 'removed_4']:

            ap4_ablation_dataframes_argmax[quadrant] = pandas.read_csv ('{0}_predictions.csv'.format(quadrant))
            ap4_ablation_dataframes_argmax[quadrant]  = ap4_ablation_dataframes_argmax[quadrant].iloc[:, 1:]
            ap4_ablation_dataframes[quadrant] = pandas.read_csv ('{0}_proba_predictions.csv'.format(quadrant))
            ap4_ablation_dataframes[quadrant]  = ap4_ablation_dataframes[quadrant].iloc[:, 1:]

            for index,row in ap4_ablation_dataframes[quadrant].iterrows():
                to_array = ap4_ablation_dataframes[quadrant].at[index,'Y_pred'].strip('[[').strip(']]').split(' ')
                to_array = list(filter(None, to_array))
                to_array = numpy.asarray(to_array).astype(numpy.float)
                ap4_ablation_dataframes[quadrant].at[index,'Y_pred'] = to_array

        #AP4_Ablations______________________________________________________________________________________________________________
        for quadrant in ['removed_1', 'removed_2', 'removed_3', 'removed_4']:

            print("Ablation for quadrant: ", quadrant)

            ablation = quadrant
            df = ap4_ablation_dataframes[ablation]
            df_argmax = ap4_ablation_dataframes_argmax[ablation]

            ablation = ablation.split('_')

            for i in range(len(ablation)):

                ablation[i] = ablation[i].capitalize()

            ablation = ' '.join(ablation)

            y_pred = df['Y_pred']
            labels_test = df['Y_test']

            y_pred_argmax = df_argmax['Y_pred'].to_numpy()

            con_matrix =  confusion_matrix(labels_test, y_pred_argmax)

            print(con_matrix)

            print(classification_report(labels_test, y_pred_argmax, target_names = ['Amyloidosis', 'HCM', 'HTN']))

            test_set_df = pandas.DataFrame(0, index = range(len(labels_test)), columns=[0, 1, 2])

            for i,j in enumerate(labels_test):

                test_set_df.loc[i, j] = 1
                
            test_set_df.rename({0: 'Amyloidosis', 1: 'HCM', 2: 'HTN'}, axis=1, inplace=True)

            y_pred_df = pandas.DataFrame()

            for i,j in enumerate(y_pred):

                new_row = {'Amyloidosis': y_pred[i][0], 'HCM': y_pred[i][1], 'HTN': y_pred[i][2]}
                y_pred_df = y_pred_df.append(new_row, ignore_index=True)
                
            Amyloidosis_AUC_LR = compute_roc(X=y_pred_df.iloc[:, 0], y=test_set_df.iloc[:, 0], pos_label=True)
            print("Amyloidosis_AUC: ROC-AUC=%.3f" % (Amyloidosis_AUC_LR.auc))
            HCM_AUC_LR = compute_roc(X=y_pred_df.iloc[:, 1], y=test_set_df.iloc[:, 1], pos_label=True)
            print("HCM_AUC: ROC-AUC=%.3f" % (HCM_AUC_LR.auc))
            HTN_AUC_LR = compute_roc(X=y_pred_df.iloc[:, 2], y=test_set_df.iloc[:, 2], pos_label=True)
            print("HTN_AUC: ROC-AUC=%.3f" % (HTN_AUC_LR.auc))

        #Load_Test_Predictions______________________________________________________________________________________________________________
        test_data_set, test_labels, temp = [],[],[]

        a_file = open(test_path + "/test_data_set.txt", "r")
        for row in a_file:
            
            row = float(row.strip())
            temp.append(row)
            if len(temp) == 18:

                test_data_set.append(temp)
                temp = []

        a_file.close()

        a_file = open(test_path + "/test_labels.txt", "r")
        for row in a_file:

            row = float(row.strip())
            test_labels.append(row)

        test_data_set, test_labels = numpy.array(test_data_set), numpy.array(test_labels)

        #Fusion_Ablations______________________________________________________________________________________________________________
        ablation_dataframes = {}
        ablation_dataframes_argmax = {}
        views = {'AP2':1, 'AP3':2, 'AP4':3, 'PSAX_V':4, 'PSAX_M':5, 'PLAX':6}

        for view in views:

            ablation_dataframes[view] = pandas.DataFrame(template)
            ablation_dataframes_argmax[view] = pandas.DataFrame(template)
            l,r = views[view]*3 - 3, views[view]*3

            for i in range(len(test_labels)):

                predictions = list(test_data_set[i])

                for j in range(l,r):

                    predictions[j] = -2

                predictions = numpy.array(predictions)
                ground_truth = test_labels[i]

                prediction = LR_model.predict_proba(predictions.reshape(1, -1))
                prediction_argmax =  LR_model.predict(predictions.reshape(1, -1))
                
                ablation_dataframes[view] = pandas.concat([pandas.DataFrame([[prediction,ground_truth]], columns=['Y_pred', 'Y_test']), ablation_dataframes[view] ], ignore_index=True)
                ablation_dataframes_argmax[view] = pandas.concat([pandas.DataFrame([[prediction_argmax[0],ground_truth]], columns=['Y_pred', 'Y_test']), ablation_dataframes_argmax[view] ], ignore_index=True)
        
        for view in views:

            ablation_dataframes[view].to_csv(dataframe_path + '/{0}_proba_obf_predictions.csv'.format(view))
            ablation_dataframes_argmax[view].to_csv(dataframe_path + '/{0}_obf_predictions.csv'.format(view))

        ablation_dataframes = {}
        ablation_dataframes_argmax = {}
        views = ['AP2', 'AP3', 'AP4', 'PSAX_V', 'PSAX_M', 'PLAX']

        for view in views:

            ablation_dataframes[view] = pandas.read_csv('./{0}_proba_obf_predictions.csv'.format(view)).iloc[:, 1:]
            ablation_dataframes_argmax[view] = pandas.read_csv('./{0}_obf_predictions.csv'.format(view)).iloc[:, 1:]

            for index,row in ablation_dataframes[view].iterrows():

                to_array = ablation_dataframes[view].at[index,'Y_pred'].strip('[[').strip(']]').split(' ')
                to_array = list(filter(None, to_array))
                to_array = numpy.asarray(to_array).astype(numpy.float)
                ablation_dataframes[view].at[index,'Y_pred'] = to_array

        for view in views:

            print("Fusion Ablation Obscuring: ", view)
            ablation = view
            df = ablation_dataframes[ablation]
            df_argmax = ablation_dataframes_argmax[ablation]

            y_pred = df['Y_pred']
            labels_test = df['Y_test']

            y_pred_argmax = df_argmax['Y_pred'].to_numpy()

            print(classification_report(labels_test, y_pred_argmax, target_names = ['Amyloidosis', 'HCM', 'HTN']))

            class_names = ['Amyloidosis', 'HCM', 'HTN']

            con_matrix =  confusion_matrix(labels_test, y_pred_argmax)
            print(con_matrix)

            test_set_df = pandas.DataFrame(0, index = range(len(labels_test)), columns=[0, 1, 2])

            for i,j in enumerate(labels_test):

                test_set_df.loc[i, j] = 1
                
            test_set_df.rename({0: 'Amyloidosis', 1: 'HCM', 2: 'HTN'}, axis=1, inplace=True)

            y_pred_df = pandas.DataFrame()

            for i,j in enumerate(y_pred):

                new_row = {'Amyloidosis': y_pred[i][0], 'HCM': y_pred[i][1], 'HTN': y_pred[i][2]}
                y_pred_df = y_pred_df.append(new_row, ignore_index=True)
                
            Amyloidosis_AUC_LR = compute_roc(X=y_pred_df.iloc[:, 0], y=test_set_df.iloc[:, 0], pos_label=True)
            print("Amyloidosis_AUC: ROC-AUC=%.3f" % (Amyloidosis_AUC_LR.auc))
            HCM_AUC_LR = compute_roc(X=y_pred_df.iloc[:, 1], y=test_set_df.iloc[:, 1], pos_label=True)
            print("HCM_AUC: ROC-AUC=%.3f" % (HCM_AUC_LR.auc))
            HTN_AUC_LR = compute_roc(X=y_pred_df.iloc[:, 2], y=test_set_df.iloc[:, 2], pos_label=True)
            print("HTN_AUC: ROC-AUC=%.3f" % (HTN_AUC_LR.auc))

    return "Finished processing ablations"

