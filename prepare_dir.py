from pathlib import Path

def create_directories():

    model_path = r'./models'
    study_path_training = r'./studies/training'
    study_path_test = r'./studies/test'
    ablation_path = r'./ablated_images'
    study_training_path = r'./temp/study_wide/training'
    study_test_path = r'./temp/study_wide/test'
    generic_training_path = r'./temp/image_wide/training'
    generic_test_path = r'./temp/image_wide/test'
    video_frame_path = r'./temp/video_frames'
    processed_frames_path = r'./temp/processed_frames'
    image_path = r'./temp/processed_frames'
    video_path = r'./temp/videos'

    for path in [model_path, study_path_training, study_path_test, ablation_path, study_training_path, study_test_path, generic_training_path, generic_test_path, video_frame_path, processed_frames_path, image_path, video_path]:

        Path(path).mkdir(parents = True, exist_ok = True)

    for disease in ["Amyloidosis", "HCM", "HTN"]:

        Path(study_path_training + "/" + disease).mkdir(parents = True, exist_ok = True)
        
    return "Finished preparing directories. Upload training studies to the studies folder"
