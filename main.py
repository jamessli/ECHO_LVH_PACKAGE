import warnings
warnings.filterwarnings("ignore")
import argparse
from tkinter import E
import numpy
import tensorflow
import shutil
import pickle
from pathlib import Path
from torch import device
import prepare_dir
import echo_video_processor
import train_generic
import view_based_training
import fusion_training
import test_model
import inference
import ablations
import overlay_generator
import test_set
import downloader
import math
import os

model_path = r'/models'
study_path_training = r'/studies/training'
study_path_test = r'/studies/test'
ablation_path = r'/ablated_images'
study_training_path = r'/temp/study_wide/training'
study_validation_path = r'/temp/study_wide/validation'
study_test_path = r'/temp/study_wide/test'
generic_training_path = r'/temp/image_wide/training'
generic_validation_path = r'/temp/image_wide/validation'
generic_test_path = r'/temp/image_wide/test'
view_training_path = r'/temp/view_wide/training'
view_validation_path = r'/temp/view_wide/validation'
view_test_path = r'/temp/view_wide/test'
video_frame_path = r'/temp/video_frames'
image_path = r'/temp/processed_frames'
overlay_path = r'/temp/overlays'
video_path = r'/videos'
results_path = r'/results'
dataframe_path = r'/temp/dataframes'

parser = argparse.ArgumentParser(description="Conduct train, test, or demo operations")
parser.add_argument('--prepare', help='prepare directories, first thing that should be run or to clear existing files')
parser.add_argument('--download', help='use in conjunction with prepare to download pre-trained models')
parser.add_argument('--train', help='prepare directories, first thing that should be run or to clear existing files')
parser.add_argument('--test', help='prepare directories, first thing that should be run or to clear existing files')
parser.add_argument('--ablate', help = 'test with ablation included')
parser.add_argument('--device', nargs = '*', help='set device(s)', default = "CPU:0")
args = parser.parse_args()

physical_devices = tensorflow.config.list_physical_devices()

for i,_ in enumerate(physical_devices):

        physical_devices[i] = str(physical_devices[i]).split('/physical_device:')[-1].split(",")[0].replace("\'", '')

def main():

    if args.device and args.device[0] == "list":

        print("Selectable Devices: ", physical_devices)

    else:

        inference_device = list(args.device.split(' '))

        if args.prepare:

            if Path(args.prepare).is_dir:

                print(setup_directory(args.prepare))

            else:

                print("Directory does not exist")

            if args.download:

                print("Download status: ", downloader.downloader(args.prepare))

        elif args.train:
            
            if args.train:

                if Path(args.train).is_dir:
                    
                    #Get Study, Disease, and View
                    print("Beginning Training on Device: ", inference_device)
                    print(process_study("training", args.train, inference_device))
                    print("Performing generic model training. This may take a while...")
                    print(perform_training(args.train, inference_device))
                    print("Performing view-wide model training. This may take a while...")
                    print(perform_training_views(args.train, inference_device))
                    print("Performing late fusion tranining. This may take a while...")
                    print(perform_training_fusion(args.train, inference_device))
                    print("Completed Deep Learning and Late Fusion Training")
                    print("Running validation on test data set")
                    print(perform_validation(args.train, inference_device))
                    print("Completed Model Validation")
                    print("Exiting...")

                else:

                    print("Directory does not exist. Exiting...")
        
        elif args.test:
            
            if not args.ablate:

                if Path(args.test).is_dir:

                    if len(list(Path(args.test + "/models/").glob('*'))) != 7:

                        print("Downloading models over network")
                        print(download_models(args.test))

                    print("Running inference on test data set")
                    print(process_study("testing", args.test, inference_device))
                    
                    print("Test Dataset Complete")

                else:

                    print("Directory does not exist")

            elif args.test and args.ablate:

                if Path(args.test).is_dir:

                    print("Generating results from test dataset")
                    print(perform_ablation(args.test, inference_device))

                else:

                    print("Directory does not exist")

            else:

                print("Directory does not exist")

    return None


def setup_directory(directory):

    prepare_dir.create_study_dir(directory)
    return prepare_dir.create_directories(directory)

def process_study(operation, directory, inference_device):
    
    if operation == "training":
        
        #PART 0: Clear existing directory
        prepare_dir.create_directories(directory)
        #PART 1: Split into View/Study/Generic data directories
        study_path = directory + study_path_training
        train_path = directory + study_training_path
        val_path = directory + study_validation_path
        test_path = directory + study_test_path

        study_count = 0

        for disease in Path(study_path).iterdir():

            study_count += len(list(disease.iterdir()))

        print("Located {0} studies. Performing preprocessing...".format(study_count))

        views = ["PSAX_V", "PSAX_M", "PLAX", "AP2", "AP3", "AP4", "A2", "A3", "A4"]

        current_count = 0
 
        for disease in Path(study_path).iterdir():

            disease_name = str(disease).split(os.sep)[-1]
            
            for study in disease.iterdir():
                
                study_name = str(study).split(os.sep)[-1]
                videos = list(study.glob("*.avi")) + list(study.glob("*.mp4"))
                
                
                if current_count%100 < 72: #Use mod 10 < 7 and mod 10< 9 for actual

                    output_path = train_path
                    img_output_path = directory + generic_training_path
                    view_output_path = directory + view_training_path

                elif current_count%100 < 90:

                    output_path = val_path
                    img_output_path = directory + generic_validation_path
                    view_output_path = directory + view_validation_path

                else:

                    output_path = test_path
                    img_output_path = directory + generic_test_path
                    view_output_path = directory + view_test_path
                    
                for video in videos:

                    for view in views:

                        if view in str(video).split(os.sep)[-1]:

                            view_name = view

                            if view_name == "A2":

                                view_name = "AP2"

                            elif view_name == "A3":

                                view_name = "AP3"

                            elif view_name == "A4":

                                view_name = "AP4"

                            break
                    
                    if not view_name:

                        return "View not supported..."

                    echo_video_processor.list_cycler(str(video), study_name, disease_name, view_name, output_path)

                    for file in list(Path("{0}/{1}/{2}/{3}".format(output_path, disease_name, study_name, view_name)).glob('*.png')):

                        filename = str(file).split(os.sep)[-1]

                        shutil.copyfile(file, Path("{0}/{1}/{2}/{3}".format(view_output_path, view_name, disease_name, filename)))
                        shutil.copyfile(file, Path("{0}/{1}/{2}".format(img_output_path, disease_name, filename)))

                print("Finished Processing: ", study_name)
                current_count += 1

        return "Finished Processing " + str(study_count) + " studies"

    elif operation == "testing":

        study_path = directory + study_path_test
        models_input_path = directory + model_path
        video_frames_output_path = directory + video_frame_path
        image_output_path = directory + image_path
        overlay_output_path = directory + overlay_path
        video_output_path = directory + video_path
        results_output_path = directory + results_path
        dataframes_output_path = directory + dataframe_path

        views = ["PSAX_V", "PSAX_M", "PLAX", "AP2", "AP3", "AP4", "A2", "A3", "A4"]
        disease_map = {0: "Amyloidosis", 1: "HCM", 2: "HTN"}

        mirrored_strategy = tensorflow.distribute.MirroredStrategy(inference_device)

        with mirrored_strategy.scope():

            AP2_model = tensorflow.keras.models.load_model(str(models_input_path) + "/AP2_classifier.h5")
            AP3_model = tensorflow.keras.models.load_model(str(models_input_path) + "/AP3_classifier.h5")
            AP4_model = tensorflow.keras.models.load_model(str(models_input_path) + "/AP4_classifier.h5")
            PSAX_V_model = tensorflow.keras.models.load_model(str(models_input_path) + "/PSAX_V_classifier.h5")
            PSAX_M_model = tensorflow.keras.models.load_model(str(models_input_path) + "/PSAX_M_classifier.h5")
            PLAX_model = tensorflow.keras.models.load_model(str(models_input_path) + "/PLAX_classifier.h5")
        
        with open(str(models_input_path) + "/late_fusion_regressor.sav", 'rb') as f:
        
            LR_model = pickle.load(open(str(models_input_path) + "/late_fusion_regressor.sav", 'rb'))
        
        model_list = {"AP2": AP2_model, "AP3": AP3_model, "AP4": AP4_model, "PSAX_V": PSAX_V_model, "PSAX_M": PSAX_M_model, "PLAX": PLAX_model}
        #model_list = {"AP2": AP2_model, "AP3": AP2_model, "AP4": AP2_model, "PSAX_V": AP2_model, "PSAX_M": AP2_model, "PLAX": AP2_model}

        fusion_input = {"AP2": numpy.array([-1, -1, -1]), "AP3": numpy.array([-1, -1, -1]), "AP4": numpy.array([-1, -1, -1]), "PSAX_V": numpy.array([-1, -1, -1]), "PSAX_M": numpy.array([-1, -1, -1]), "PLAX": numpy.array([-1, -1, -1])}
        test_vector = None

        for study in Path(study_path).iterdir():
            
            study_name = str(study).split("/")[-1]
            print("Currently Processing Study: ", study_name)
            
            videos = list(study.glob("*.mp4")) + list(study.glob("*.avi"))

            for video in videos:

                for view in views:

                    if view in str(video).split('/')[-1]:

                        view_name = view

                        if view_name == "A2":

                            view_name = "AP2"

                        elif view_name == "A3":

                            view_name = "AP3"

                        elif view_name == "A4":

                            view_name = "AP4"

                        break
                
                if not view_name:

                    return "View not supported..."

                raw_path = "{0}/{1}/{2}".format(video_frames_output_path, study_name, view_name)
                fps = echo_video_processor.study_splitter(str(video), study_name, raw_path)
                processed_path = "{0}/{1}/{2}".format(image_output_path, study_name, view_name)
                echo_video_processor.list_cycler(str(video), study_name, "test", view_name, processed_path)
                print("Generating Heatmaps and Videos")

                print(raw_path, processed_path)
                pred = overlay_generator.load_video(raw_path, processed_path, overlay_output_path, video_output_path, dataframes_output_path, view_name, study_name, model_list[view_name], fps)
                fusion_input[view_name] = pred[0]

            for key in fusion_input:

                if test_vector is None:

                    test_vector = fusion_input[key]

                else:

                    test_vector = numpy.concatenate((test_vector, fusion_input[key]))

            fusion_prediction = perform_testing(test_vector, LR_model)

            print(study_name, "\n____________________________________________")
            print("Predicted Disease: ", disease_map[fusion_prediction[1][0]])
            print("Amyloidosis Confidency: ", fusion_prediction[0][0][0])
            print("HCM Confidency: ", fusion_prediction[0][0][1])
            print("HTN Confidency: ", fusion_prediction[0][0][2])
            print(" \n____________________________________________")


def perform_training(directory, devices):

    training_path = directory + generic_training_path
    validation_path = directory + generic_validation_path
    output_path = directory + model_path

    return train_generic.train_generic(training_path, validation_path, output_path, devices)

def perform_training_views(directory, devices):

    training_path = directory + view_training_path
    validation_path = directory + view_validation_path
    output_path = directory + model_path

    return view_based_training.train_views(training_path, validation_path, output_path, devices)

def perform_training_fusion(directory, devices):

    training_path = directory + study_training_path
    validation_path = directory + study_validation_path
    output_path = directory + model_path

    return fusion_training.train_fusion(training_path, validation_path, output_path, devices)

def perform_validation(directory, devices):

    test_path = directory + study_test_path
    output_path = directory + model_path

    return inference.validate_test(test_path, output_path, devices)

def perform_testing(test_vector, LR_model):
    
    return test_set.run_testing(test_vector, LR_model)

def download_models(directory):

    models = directory + model_path
    return downloader.downloader(models)

def perform_ablation(directory, devices):

    study_path = directory + study_test_path
    models_input_path = directory + model_path
    ablated_path = directory + ablation_path
    image_output_path = directory + image_path
    dataframes_output_path = directory + dataframe_path
    video_frames_output_path = directory + video_frame_path

    views = ["PSAX_V", "PSAX_M", "PLAX", "AP2", "AP3", "AP4", "A2", "A3", "A4"]

    for study in Path(study_path).iterdir():
        
        study_name = str(study).split(os.sep)[-1]
        print("Currently Processing Study: ", study_name)
        
        videos = list(study.glob("*.mp4")) + list(study.glob("*.avi"))

        for video in videos:

            for view in views:

                if view in str(video).split(os.sep)[-1]:

                    view_name = view

                    if view_name == "A2":

                        view_name = "AP2"

                    elif view_name == "A3":

                        view_name = "AP3"

                    elif view_name == "A4":

                        view_name = "AP4"

                    break
            
            if not view_name:

                return "View not supported..."

            raw_path = "{0}/{1}/{2}".format(video_frames_output_path, study_name, view_name)
            fps = echo_video_processor.study_splitter(str(video), study_name, raw_path)
            processed_path = "{0}/{1}/{2}".format(image_output_path, study_name, view_name)
            echo_video_processor.list_cycler(str(video), study_name, "test", view_name, processed_path)

    print(study_path, ablated_path, models_input_path, dataframes_output_path, image_output_path, devices)
    ablations.generate_ablated(study_path, ablated_path, models_input_path, dataframes_output_path, image_output_path, devices)
if __name__ == '__main__':

    main()


