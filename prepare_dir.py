from pathlib import Path

def create_directories(directory):

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
    processed_frames_path = r'./temp/processed_frames'
    image_path = r'/temp/processed_frames'
    video_path = r'/videos'
    results_path = r'/results'
    dataframe_path = r'/temp/dataframes'

    for path in [model_path, study_path_test, ablation_path, study_training_path, study_validation_path, study_test_path, video_frame_path, processed_frames_path, image_path, video_path, results_path, dataframe_path]:

        path = "{0}{1}".format(directory, path)
        Path(path).mkdir(parents = True, exist_ok = True)

    for disease in ["Amyloidosis", "HCM", "HTN"]:

        Path("{0}{1}/{2}".format(directory, study_path_training, disease)).mkdir(parents = True, exist_ok = True)
        Path("{0}{1}/{2}".format(directory, generic_training_path, disease)).mkdir(parents = True, exist_ok = True)
        Path("{0}{1}/{2}".format(directory, generic_validation_path, disease)).mkdir(parents = True, exist_ok = True)
        Path("{0}{1}/{2}".format(directory, generic_test_path, disease)).mkdir(parents = True, exist_ok = True)

    
    for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

        for disease in ["Amyloidosis", "HCM", "HTN"]:

            Path("{0}{1}/{2}/{3}".format(directory, view_training_path, view, disease)).mkdir(parents = True, exist_ok = True)
            Path("{0}{1}/{2}/{3}".format(directory, view_validation_path, view, disease)).mkdir(parents = True, exist_ok = True)
            Path("{0}{1}/{2}/{3}".format(directory, view_test_path, view, disease)).mkdir(parents = True, exist_ok = True)

    return "Finished preparing [ {0} ] for use. Upload training studies to the studies folder".format(directory)
