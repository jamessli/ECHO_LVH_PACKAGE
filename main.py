import sys
import logging
import pathlib
import argparse
import tensorflow
import shutil

from pathlib import Path
import prepare_dir
import echo_video_processor
import train_generic
import training_and_fusion
'''
import overlay_generator
import train_fusion
import test_fusion
import test_ablation
'''

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
video_path = r'/temp/videos'

parser = argparse.ArgumentParser(description="Conduct train, test, or demo operations")
parser.add_argument('--prepare', help='prepare directories, first thing that should be run or to clear existing files')
parser.add_argument('--train', help='prepare directories, first thing that should be run or to clear existing files')
parser.add_argument('--test', help='prepare directories, first thing that should be run or to clear existing files')
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
                    print(perform_training(args.train, inference_device))

                else:

                    print("Directory does not exist")
        
        elif args.test:
            
            #Get Study, Disease, and View
            print("Beginning Testing on Device: ", inference_device)
            print(process_study("testing"))

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

    studies_path = directory + generic_training_path
    output_path = directory + model_path

    return train_generic.train_generic(studies_path, output_path, devices)

def perform_training_fusion(studies_path, output_path, devices):

    print( train_generic.train_generic(studies_path, output_path, devices))
    return training_and_fusion.train_fusion(studies_path, output_path, devices)

def perform_validation():

    return None


if __name__ == '__main__':

    main()




