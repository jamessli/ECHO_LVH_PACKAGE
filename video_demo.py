from isort import file
import streamlit as st
import pandas as pandas
from IPython.display import display
from pathlib import Path
from urllib.error import URLError
import time
import sys
import echo_video_processor
import overlay_generator
import tensorflow as tensorflow
import numpy as numpy
from tensorflow import keras
import pickle
import time


try:
    
    main_path = r'./study/'
    model_path = r'./models/'

    #st.set_page_config(layout="wide")

    study_name = st.text_input("Enter File Path", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    file_path = main_path + '/' + study_name
    disease_map = {0: "Amyloidosis", 1: "HCM", 2: "HTN"}

    physical_devices = tensorflow.config.list_physical_devices()
    for i,_ in enumerate(physical_devices):

        physical_devices[i] = str(physical_devices[i]).split('/physical_device:')[-1].split(",")[0].replace("\'", '')
    
    inference_device = st.selectbox('Select Inference Device', physical_devices)

    st.markdown("Valid filepath: " + str(Path(file_path).is_dir()))

    if st.button ('Run on Directory') and Path(file_path).is_dir() :

        #Get directories
        views = []
        
        for subdir in Path(file_path).iterdir():

            if subdir.is_dir():

                view = str(subdir).split("\\")[-1]
                views.append(subdir)
                model_dict = {}

                with st.spinner('Processing views and generating frames, this may take some time...'):

                    views,fps = echo_video_processor.list_cycler(study_name)
                    st.info("Processed the following views: " + ' '.join(views))

                with st.spinner('Loading necessary view models...'):

                    with tensorflow.device(inference_device):

                        for view in views:

                            if view == "AP2":
                                AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/AP2_classifier.h5")
                                model_dict['AP2'] = AP2_model
                                st.info("Successfully Loaded AP2 Classifier")

                            if view == "AP3":

                                AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/AP3_classifier.h5")
                                model_dict['AP3'] = AP3_model
                                st.info("Successfully Loaded AP3 Classifier")

                            if view == "AP4":

                                AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/AP4_classifier.h5")
                                model_dict['AP4'] = AP4_model
                                st.info("Successfully Loaded AP4 Classifier")
                        
                            if view == "PSAX_V":

                                    PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_V_classifier.h5")
                                    model_dict['PSAX_V'] = PSAX_V_model
                                    st.info("Successfully Loaded PSAX_V Classifier")

                            if view == "PSAX_M":

                                    PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/PSAX_M_classifier.h5")
                                    model_dict['PSAX_M'] = PSAX_M_model
                                    st.info("Successfully Loaded PSAX_V Classifier")
                        
                            if view == "PLAX":

                                PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/PLAX_classifier.h5")
                                model_dict['PLAX'] = PLAX_model
                                st.info("Successfully Loaded PSAX_V Classifier")

                            with open(str(model_path) + '/late_fusion_regressor.sav', 'rb') as input_file:

                                LR_model = pickle.load(input_file)
                                model_dict['Fusion'] = LR_model

                with st.spinner('View: {0}: Running predictions and generating videos'.format(view)):

                    fusion_vector = []

                    for i,view in enumerate(['AP2', 'AP3', 'AP4', 'PSAX_V', 'PSAX_M', 'PLAX']):

                        if views[i] is not view:

                            views.insert(i, "Empty")

                        predictions = overlay_generator.load_study(study_name, views[i], model_dict[view], AP2_model, fps)

                        for prediction in predictions:

                            fusion_vector.append(prediction)

                        if views[i] != "Empty":
                        
                            v = views[i]
                            
                            vid_file = list(Path('./output/{0}'.format(study_name)).glob("*{0}.mp4".format(v)))
                            vid_file =  str(vid_file[0])

                            st.header("{0}: {1}".format(study_name, v))

                            with open(vid_file, 'rb') as video:
                                
                                st.write(vid_file.split('/')[-1])
                                st.video(video, format="video/mp4", start_time=0)

                            st.markdown("Amyloidosis: {0}   HCM: {1}   HTN: {2}".format(prediction[0], prediction[1], prediction[2]))

                fusion_vector = numpy.array(fusion_vector)
                prediction = LR_model.predict_proba(fusion_vector.reshape(1, -1))

                st.title("Study-Wide Prediction")
                st.markdown("Amyloidosis: {0}   HCM: {1}   HTN: {2}".format(prediction[0][0], prediction[0][1], prediction[0][2]))
                st.markdown("Prediction Class: {0}".format(disease_map[numpy.argmax(prediction)]))

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )