import sys
import logging
import pathlib
import argparse
import tensorflow

from pathlib import Path
import prepare_dir
import echo_video_processor
import train_generic
import training_and_fusion
import seperate_studies
'''
import overlay_generator
import train_fusion
import test_fusion
import test_ablation
'''

model_path = r'./models'
study_path_training = r'./studies/training'
study_path_test = r'./studies/test'
ablation_path = r'./ablated_images'
study_training_path = r'./temp/study_wide/training'
study_validation_path = r'./temp/study_wide/validation'
study_test_path = r'./temp/study_wide/test'
generic_training_path = r'./temp/image_wide/training'
generic_validation_path = r'./temp/image_wide/validation'
generic_test_path = r'./temp/image_wide/test'
video_frame_path = r'./temp/video_frames'
processed_frames_path = r'./temp/processed_frames'
image_path = r'./temp/processed_frames'
video_path = r'./temp/videos'

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

            print(args.prepare)
            print(setup_directory())

        elif args.train:
            
            #Get Study, Disease, and View
            print("Beginning Training on Device: ", inference_device)
            print(process_study("training"))
            print(perform_training(args.train, args.train, inference_device))
        
        elif args.test:
            
            #Get Study, Disease, and View
            print("Beginning Testing on Device: ", inference_device)
            print(process_study("testing"))

def setup_directory():

    return prepare_dir.create_directories()

def process_study(operation):

    if operation == "training":

        study_path = study_path_training

        study_count = 0

        for disease in Path(study_path).iterdir():

            study_count += len(list(disease.iterdir()))

        for disease in Path(study_path).iterdir():

            disease_name = str(disease).split('/')[-1]
            current_count = 0

            for study in disease.iterdir():
                
                study_name = str(study).split('/')[-1]
                videos = list(study.glob("*.avi")) + list(study.glob("*.mp4"))

                if current_count < round(study_count*.15)/3:

                    output_path = study_training_path

                elif current_count < round(study_count*.35)/3:

                    output_path = study_validation_path

                else:

                    output_path = study_test_path

                for video in videos:

                    view_name = str(video).split('/')[-1].split('_', 1)[1].split('.')[0].strip().upper()

                    if view_name == "A2" or view_name == "A3" or view_name == "A4":

                        view_name = view_name[0] + "P" + view_name[1]

                    if view_name == "PSAX_V1":

                        view_name = "PSAX_V"

                    print("ECHO", echo_video_processor.list_cycler(str(video), study_name, disease_name, view_name, output_path, operation))

        return "Finished Processing " + str(study_count) + " studies"

    #TODO TESTING

def perform_training(studies_path, output_path, devices):

    return train_generic.train_generic(studies_path, output_path, devices)

def perform_training_fusion(studies_path, output_path, devices):

    return train_generic.train_generic(studies_path, output_path, devices)

def perform_validation():

    return None

if __name__ == '__main__':

    main()




