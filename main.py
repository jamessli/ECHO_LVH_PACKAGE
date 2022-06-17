from pyexpat import model
import sys
import logging
import pathlib
import argparse
import tensorflow
import shutil

from pathlib import Path

from torch import device
import prepare_dir
import echo_video_processor
import train_generic
import view_based_training
import fusion_training
import test_model
#import inference
import ablation
import overlay_generator
import downloader


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
processed_frames_path = r'/temp/processed_frames'
image_path = r'/temp/processed_frames'
video_path = r'/videos'
results_path = r'/results'
dataframe_path = r'/temp/dataframes'

parser = argparse.ArgumentParser(description="Conduct train, test, or demo operations")
parser.add_argument('--prepare', help='prepare directories, first thing that should be run or to clear existing files')
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

        inference_device = args.device

        if args.prepare:

            if Path(args.prepare).is_dir:

                print(setup_directory(args.prepare))

            else:

                print("Directory does not exist")

        elif args.train:
            
            if args.train:

                if Path(args.train).is_dir:
                    
                    #Get Study, Disease, and View
                    print("Beginning Training on Device: ", inference_device)
                    print(process_study("training", args.train))
                    print("Performing generic model training. This may take a while...")
                    print(perform_training(args.train, inference_device))
                    print("Performing view-wide model training. This may take a while...")
                    print(perform_training_views(args.train, inference_device))
                    print("Performing late fusion tranining. This may take a while...")
                    print(perform_training_fusion(args.train, inference_device))
                    print("Completed Training")

                else:

                    print("Directory does not exist. Exiting...")
        
        elif args.test:
            
            if args.test and not args.ablate:

                if Path(args.test).is_dir:

                    print("Generating results from test dataset")
                    print(perform_validation(args.test, inference_device))
                    print("Test Dataset Complete")

                else:

                    print("Directory does not exist")

            else:

                if Path(args.test).is_dir:

                    print("Generating results from test dataset")
                    print(perform_ablation(args.test, inference_device))

                else:

                    print("Directory does not exist")

    return None


def setup_directory(directory):

    return prepare_dir.create_directories(directory)

def process_study(operation, directory):
    
    if operation == "training":
        
        #PART 1: Split into View/Study/Generic data directories
        study_path = directory + study_path_training
        train_path = directory + study_training_path
        val_path = directory + study_validation_path
        test_path = directory + study_test_path

        study_count = 0

        for disease in Path(study_path).iterdir():

            study_count += len(list(disease.iterdir()))

        print("Located {0} studies. Performing preprocessing...".format(study_count))
 
        for disease in Path(study_path).iterdir():

            disease_name = str(disease).split('/')[-1]
            current_count = 0

            for study in disease.iterdir():
                
                study_name = str(study).split('/')[-1]
                videos = list(study.glob("*.avi")) + list(study.glob("*.mp4"))

                if current_count < round(study_count*.15)/3:

                    output_path = train_path
                    img_output_path = directory + generic_training_path
                    view_output_path = directory + view_training_path

                elif current_count < round(study_count*.35)/3:

                    output_path = val_path
                    img_output_path = directory + generic_validation_path
                    view_output_path = directory + view_validation_path

                else:

                    output_path = test_path
                    img_output_path = directory + generic_test_path
                    view_output_path = directory + view_test_path

                for video in videos:

                    view_name = str(video).split('/')[-1].split('_', 1)[1].split('.')[0].strip().upper()

                    if view_name == "A2" or view_name == "A3" or view_name == "A4":

                        view_name = view_name[0] + "P" + view_name[1]

                    if view_name == "PSAX_V1":

                        view_name = "PSAX_V"

                    print(echo_video_processor.list_cycler(str(video), study_name, disease_name, view_name, output_path, operation))

                    for file in list(Path("{0}/{1}/{2}/{3}".format(output_path, disease_name, study_name, view_name)).glob('*.png')):

                        filename = str(file).split('/')[-1]
                        
                        shutil.copyfile(file, "{0}/{1}/{2}/{3}".format(view_output_path, view_name, disease_name, filename))
                        shutil.copyfile(file, "{0}/{1}/{2}".format(img_output_path, disease_name, filename))

        return "Finished Processing " + str(study_count) + " studies"

    #TODO TESTING

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
    output_path = directory + results_path

    return fusion_training.train_fusion(test_path, output_path, devices)

def perform_ablation(directory, devices):

    ablations = directory + ablation_path
    test_path = directory + study_test_path
    models = directory + model_path
    dataframes = directory + dataframe_path

    return ablation.generate_ablated(test_path, ablations, models, dataframes, devices)

if __name__ == '__main__':

    main()


