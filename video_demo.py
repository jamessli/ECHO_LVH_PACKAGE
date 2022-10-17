from msilib.schema import Directory
from git import base
from isort import file
from sklearn import model_selection
import streamlit as st
import pandas as pandas
from IPython.display import display
from pathlib import Path
from urllib.error import URLError
import echo_video_processor
import overlay_generator
import tensorflow as tensorflow
import numpy as numpy
from tensorflow import keras
import pickle
import downloader
import os

physical_devices = tensorflow.config.list_physical_devices()

for i,_ in enumerate(physical_devices):

        physical_devices[i] = str(physical_devices[i]).split('/physical_device:')[-1].split(",")[0].replace("\'", '')

try:
    
    base_dir = str(Path().resolve()) + "/streamlit/"

    if not Path(base_dir).is_dir():
        
        Path(base_dir).mkdir(parents = False, exist_ok = False)

    st.info("Operating out of directory: {0\nPlease upload your study here.".format(base_dir))

    study_name = st.text_input("Enter Study Name", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    study_path = base_dir + study_name
    model_path = base_dir + '/models/'
    output_path = study_path + '/temp/overlays/'
    frames_path = study_path + '/temp/frames/'
    video_path = study_path + '/temp/new_videos'

    for new_path in [output_path, video_path, frames_path]:

        Path(new_path).mkdir(parents = False, exist_ok = False)

    if (not Path(model_path).is_dir()) or (len(list(Path(model_path).glob('*'))) != 7):

        st.info("Incomplete model set detected. Downloading models over network. This may take some time.")
        Path(model_path).mkdir(parents = False, exist_ok = False)
        st.info(downloader.downloader(model_path))

    st.info("Inferencing ready. Please select inference device from drop down menu.")

    disease_map = {0: "Amyloidosis", 1: "HCM", 2: "HTN"}

    physical_devices = tensorflow.config.list_physical_devices()
    for i,_ in enumerate(physical_devices):

        physical_devices[i] = str(physical_devices[i]).split('/physical_device:')[-1].split(",")[0].replace("\'", '')
    
    inference_device = st.selectbox('Select Inference Device', physical_devices)

    st.markdown("Valid filepath: " + str(Path(study_path).is_dir()))

    if st.button ('Run on Directory') and Path(study_path).is_dir() :

        views = {}
        model_dict = {}

        for vid_input in list(Path(study_path)).glob("R*.*"):

            file_name = str(vid_input).split(os.sep)[-1]
            view_name = file_name.split('.')[0].split('_', 1)[1]

            with st.spinner('Processing views and generating frames, this may take some time...'):

                fps = echo_video_processor.study_splitter(str(vid_input), study_name, frames_path)
                views[view_name] = fps

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

        with st.spinner("Running predictions and generating videos..."):

            predictions = {}
            fusion_vector = []

            for view in ['AP2', 'AP3', 'AP4', 'PSAX_M', 'PSAX_V', 'PLAX']:

                if view not in views:

                    fusion_vector.append[-2, -2, -2]

                else:
                    
                    prediction = overlay_generator.load_study(study_name, view, model_dict[view], model_dict[view], views[view], output_path, video_path)
                    fusion_vector.append(prediction)
                    predictions[view] = prediction

            for view in views:
                
                vid_file = list(Path(video_path + '/{0}'.format(study_name)).glob("*{0}.mp4".format(view)))
                vid_file =  str(vid_file[0])

                st.header("{0}: {1}".format(study_name, view))

                with open(vid_file, 'rb') as video:
                    
                    st.write(vid_file.split('/')[-1])
                    st.video(video, format="video/mp4", start_time=0)

                st.markdown("Amyloidosis: {0}   HCM: {1}   HTN: {2}".format(predictions[view][0], predictions[view][1], predictions[view][2]))

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