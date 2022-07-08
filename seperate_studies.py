#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas
from pathlib import Path
import shutil
from natsort import natsorted
from roc_utils import *

# In[2]:


def setup_studies(study_path):

    images_path ="{0}/{1}".format(study_path, "temp/images_path/")
    validation_path ="{0}/{1}".format(study_path, "studies/study_wise_validation/")
    training_path ="{0}/{1}".format(study_path, "studies/study_wise_training/")
    training_split_path ="{0}/{1}".format(study_path, "studies/training/")
    training_path_generic = "{0}/{1}".format(study_path, "studies/images_training/")
    validation_path_generic = "{0}/{1}".format(study_path, "studies/images_validation/")
    training_path_views = "{0}/{1}".format(study_path, "studies/view_wise/training")
    validation_path_views = "{0}/{1}".format(study_path, "studies/view_wise/validation")
    training_path_views = "{0}/{1}".format(study_path, "images_training")

    amy = set()
    hcm = set()
    htn = set()
    file_counter = 0
    study_counter = 0

    for i,set_name in zip(["Amyloidosis", "HCM", "HTN"], [amy, hcm, htn]):

        dir_path = Path(images_path + i)
        directory = list(dir_path.glob('**/*.png'))

        for file in directory:

            name = str(file).replace("-", "_")

            file_counter += 1
            filename = str(name).split('/')[-1]
            study = str(name).split('/')[-1].split('_')[0]
            view = str(name).split('/')[-1].split('_', 1)[1].split('.')[0].strip().upper()

            if view == "A2" or view == "A3" or view == "A4":

                view = view[0] + "P" + view[1]

            if view == "PSAX_V1":

                view = "PSAX_V"

            if study not in set_name:

                set_name.add(study)
                study_counter += 1
        
        print(dir_path)

    studydict = {"Amyloidosis": [], "HCM": [], "HTN": []}
    counter = 0

    for i in ["Amyloidosis", "HCM", "HTN"]:

        dir_path = Path(images_path + i)
        directory = list(dir_path.glob('**/*.png'))

        for file in directory:

            name = str(file).replace("-", "_")

            counter += 1
            filename = str(name).split('/')[-1]
            study = str(name).split('/')[-1].split('_')[0]
            view = str(name).split('/')[-1].split('_', 1)[1].split('.')[0].strip().upper()

            if view == "A2" or view == "A3" or view == "A4":

                view = view[0] + "P" + view[1]

            if view == "PSAX_V1":

                view = "PSAX_V"

            if study not in studydict[i]:

                studydict[i].append(study)
        
        studydict[i].append(counter)
        print(dir_path, counter)
        counter = 0
    
    unique_studies = set()
    counter = 0

    for i in ["Amyloidosis", "HCM", "HTN"]:

        dir_path = Path(images_path + i)
        directory = list(dir_path.glob('**/*.png'))

        for file in directory:

            name = str(file).replace("-", "_")

            counter += 1
            filename = str(name).split('/')[-1]
            study = str(name).split('/')[-1].split('_')[0]
            view = str(name).split('/')[-1].split('_', 1)[1].split('.')[0].strip().upper()

            if view == "A2" or view == "A3" or view == "A4":

                view = view[0] + "P" + view[1]

            if view == "PSAX_V1":

                view = "PSAX_V"

            if study not in unique_studies:

                unique_studies.add(study)
                new_path = study_path + study
                Path(new_path).mkdir(parents=True, exist_ok = True)

                for j in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

                    for k in ["Amyloidosis", "HCM", "HTN"]:

                        Path(new_path + "/{0}/{1}".format(j, k)).mkdir(parents = True, exist_ok = True)

            shutil.copyfile(file, study_path + "{0}/{1}/{2}/{3}".format(study, view, i, filename))
        
        print(dir_path, counter)
        counter = 0
        

    dataframe_template = {'Study':[], 'View':[], "Disease":[], "Count":[]}
    studywise_df = pandas.DataFrame(data = dataframe_template)
    studywise_df.astype({'Count': 'int32'}).dtypes
    studywise_df

    complete_studies = set()

    for study in Path(study_path).iterdir():

        study_name = str(study).split('/')[-1]
        complete_studies.add(study_name)

        for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

            for disease in ["Amyloidosis", "HCM", "HTN"]:

                #print(study_name, view, disease, len(list(Path(str(study) + "/{0}/{1}".format(view, disease)).glob('**/*'))))
                if len(list(Path(str(study) + "/{0}/{1}".format(view, disease)).glob('**/*'))) == 0 and study_name in complete_studies:

                    complete_studies.remove(study_name)

                new_row = {'Study': study_name, 'View': view, "Disease": disease, "Count": len(list(Path(str(study) + "/{0}/{1}".format(view, disease)).glob('**/*')))}
                studywise_df = studywise_df.append(new_row, ignore_index = True)

    viable_studies = []
    for study in complete_studies:

        temp_df = studywise_df.loc[ (studywise_df['Study'] == study) ]
        viable_studies.append((study, temp_df.loc[temp_df['Count'].idxmin()]['Count']))

    viable_studies.sort(key=lambda x: x[1], reverse = True)

    print(viable_studies[:16])
    validation_studies = []
    for i in viable_studies[:16]:

        validation_studies.append(i)


    for study in validation_studies:

        shutil.copytree(study_path + study[0], validation_path + "grouped_data/{0}".format(study[0]))

    min_files = []
    for study in validation_studies:

        study_name = study[0]
        count = int(study[1])

        for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

            for disease in ["Amyloidosis", "HCM", "HTN"]:

                if not Path(validation_path + "grouped_data/{0}/{1}/{2}/".format(disease, study_name, view)).exists():

                    Path(validation_path + "grouped_data/{0}/{1}/{2}/".format(disease, study_name, view)).mkdir(parents = True, exist_ok = True)

                for file in Path(validation_path + "all_complete_studies/{0}/{1}/{2}/".format(study_name, view, disease)).glob('**/*'):

                    min_files.append(str(file).split('/')[-1])

                min_files = natsorted(min_files)

                for i in range(count):

                    shutil.copy(validation_path + "all_complete_studies/{0}/{1}/{2}/{3}".format(study_name, view, disease, min_files[i]), validation_path + "grouped_data/{0}/{1}/{2}/{3}".format(disease, study_name, view, min_files[i]))

                min_files = []


    validation_study_set = set()
    for study in validation_studies:

        validation_study_set.add(study[0])


    training_data_list = {"Amyloidosis":[], "HCM":[], "HTN":[]}
    validation_data_list = {"Amyloidosis":[], "HCM":[], "HTN":[]}
    for key in studydict:

        for study in studydict[key][:-1]:

            if study not in validation_study_set:

                training_data_list[key].append(study)

            else:

                validation_data_list[key].append(study)



    training_data_list_split = {"Amyloidosis":[], "HCM":[], "HTN":[]}

    for key in training_data_list:

        while len(training_data_list_split[key]) < 30:

            training_data_list_split[key].append(training_data_list[key].pop())

    print(len(training_data_list['Amyloidosis']), len(training_data_list_split['Amyloidosis']) ,len(validation_data_list['Amyloidosis']))
    print(len(training_data_list['HCM']), len(training_data_list_split['Amyloidosis']), len(validation_data_list['HCM']))
    print(len(training_data_list['HTN']), len(training_data_list_split['Amyloidosis']), len(validation_data_list['HTN']))


    counter = 0
    for disease in training_data_list:

        if not Path(training_path_generic + "/{0}".format(disease)).exists():

                Path(training_path_generic + "/{0}".format(disease)).mkdir(parents = True, exist_ok = True)

        for study in training_data_list[disease]:

            for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:
                
                for file in Path(study_path + "/{0}/{1}/{2}/".format(study, view, disease)).glob('**/*'):

                    filename = str(file).split('/')[-1]
                    shutil.copy(study_path + "/{0}/{1}/{2}/{3}".format(study, view, disease, filename), training_path_generic + "/{0}/{1}".format(disease, view + "__" + filename))
                    counter += 1

        print(disease, counter)
        counter = 0


    # In[ ]:


    #split generic training data


    # In[45]:


    counter = 0
    for disease in training_data_list_split:

        if not Path(training_split_path + "/training_data_generic_val/{0}".format(disease)).exists():

                Path(training_split_path + "/training_data_generic_val/{0}".format(disease)).mkdir(parents = True, exist_ok = True)

        for study in training_data_list_split[disease]:

            for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:
                
                for file in Path(study_path + "/{0}/{1}/{2}/".format(study, view, disease)).glob('**/*'):

                    filename = str(file).split('/')[-1]
                    shutil.copy(study_path + "/{0}/{1}/{2}/{3}".format(study, view, disease, filename), training_split_path + "/training_data_generic_val/{0}/{1}".format(disease, view + "__" + filename))
                    counter += 1

        print(disease, counter)
        counter = 0


    # In[ ]:


    #Create generic validation set


    # In[33]:


    counter = 0
    for disease in validation_data_list:

        if not Path(validation_path_generic + "/{0}".format(disease)).exists():

                Path(validation_path_generic + "/{0}".format(disease)).mkdir(parents = True, exist_ok = True)

        for study in validation_data_list[disease]:

            for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:
                
                for file in Path(study_path + "/{0}/{1}/{2}/".format(study, view, disease)).glob('**/*'):

                    filename = str(file).split('/')[-1]
                    shutil.copy(study_path + "/{0}/{1}/{2}/{3}".format(study, view, disease, filename), validation_path_generic + "/{0}/{1}".format(disease, view + "__" + filename))
                    counter += 1

        print(disease, counter)
        counter = 0


    view_file_counter_training = {"AP2":0, "AP3":0, "AP4":0, "PSAX_V":0, "PSAX_M":0, "PLAX":0}

    for view in view_file_counter_training:

        for disease in training_data_list:

            Path(training_path_views + "/{0}/{1}/".format(view, disease)).mkdir(parents = True, exist_ok = True)

    for disease in training_data_list:

        for file_path in list(Path(training_path_generic + '/{0}'.format(disease)).glob('**/*')):

            file_path = str(file_path)
            file = file_path.split('/')[-1]
            view = file.split('__')[0]

            shutil.copy(file_path, training_path_views + "/{0}/{1}/{2}".format(view, disease, file))
            view_file_counter_training[view] += 1


    view_file_counter_validation = {"AP2":0, "AP3":0, "AP4":0, "PSAX_V":0, "PSAX_M":0, "PLAX":0}

    for view in view_file_counter_validation:

        for disease in validation_data_list:

            Path(validation_path_views + "/{0}/{1}/".format(view, disease)).mkdir(parents = True, exist_ok = True)

    for disease in validation_data_list:

        for file_path in list(Path(validation_path_generic + '/{0}'.format(disease)).glob('**/*')):

            file_path = str(file_path)
            file = file_path.split('/')[-1]
            view = file.split('__')[0]

            shutil.copy(file_path, validation_path_views + "/{0}/{1}/{2}".format(view, disease, file))
            view_file_counter_validation[view] += 1


    for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

        for disease in ["Amyloidosis", "HCM", "HTN"]:

            if not Path(training_path + "/training_data_generic/{0}/".format(disease)).exists():

                Path(training_path + "/training_data_generic/{0}/".format(disease)).mkdir(parents = True, exist_ok = True)

            for file in Path(training_path + "/training_data/{0}/{1}/".format(view, disease)).glob('**/*'):

                filename = str(file).split('/')[-1]

                shutil.copy(training_path + "/training_data/{0}/{1}/{2}".format(view, disease, filename), training_path + "/training_data_generic/{0}/{1}".format(disease, filename))


    sum = 0
    for study in validation_studies:

        sum += int(study[1])*3*6

    print(sum)

    #This equals the min nunmber of frames possible for each study * number of studies for the total size of the validation dataset. Value is 16056. Running find . -type f | wc -l in terminal yields 16056. PERFECT
    #These images are split into 3 classes: AMY, HCM, HTM and each study may not have the same number of slices; however, total number of class slices are equal. 5352 slices per class.



    #Get prediction from each model on Validation Data Set
    #For each study: 18 by N predictions where N is number of Frames and the 18 predictions are from each view: [AP2[p1,p2,p3], AP3[p1,p2,p3], AP4[p1,p2,p3], PSAX_V[p1,p2,p3], PSAX_M[p1,p2,p3], PLAX[p1,p2,p3]] x 66 Frames (example)
    #We have a Ground Truth Label for each study ( 0, 1, 2) (Since each study can only fit into one class)
    #Generate Multiclass classifaction fusion model amalgamating the prediction vector to get the per study prediction
    #48 studies, 48 fusion model predictions
    #NOTE THAT Amyloidosis/R001 != HCM/R001! Studies share names but are distinct, classes do not overlap


