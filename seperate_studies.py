#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas
import numpy
import matplotlib as mpl
import matplotlib.pyplot as pyplot
from pathlib import Path
import shutil
from natsort import natsorted
import PIL


#Tensorflow
import tensorflow
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
import itertools
from roc_utils import *

# In[2]:

'''
study_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise/'
images_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/images/training_data_new/'
validation_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_validation/'
training_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_training/'
results_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_results'
training_split_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/training/'
training_path_generic = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/training/training_data_generic'
validation_path_generic = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/training/validation_data_generic'
training_path_views = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/training/training_data_view'
validation_path_views = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/training/validation_data_view'
'''

# In[3]:


#Get Study Number and Total File Number


# In[4]:


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
    


# In[5]:


#Find count of studies in each bin


# In[6]:


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
    


# In[ ]:


#Seperate into studies


# In[ ]:


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
    


# In[7]:


dataframe_template = {'Study':[], 'View':[], "Disease":[], "Count":[]}
studywise_df = pandas.DataFrame(data = dataframe_template)
studywise_df.astype({'Count': 'int32'}).dtypes
studywise_df


# In[ ]:


#Find Complete Sets


# In[8]:


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


# In[8]:


complete_studies


# In[9]:


studywise_df


# In[10]:


viable_studies = []
for study in complete_studies:

    temp_df = studywise_df.loc[ (studywise_df['Study'] == study) ]
    viable_studies.append((study, temp_df.loc[temp_df['Count'].idxmin()]['Count']))

viable_studies.sort(key=lambda x: x[1], reverse = True)


# In[ ]:


#Get Top 16 shared studies (48 total (NOTE THEY"RE ALL DIFFERENT))


# In[11]:


print(viable_studies[:16])
validation_studies = []
for i in viable_studies[:16]:

    validation_studies.append(i)


# In[ ]:


for study in validation_studies:

    shutil.copytree(study_path + study[0], validation_path + "grouped_data/{0}".format(study[0]))


# In[ ]:


#There are a total of 167 studies available for our use currently. Of these, .1 (16 rounded down) are used for our validation set, total 22182 images BEFORE reduction. NOTE, each of these validation studies are Complete Studies


# In[12]:


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

    


# In[ ]:


#Create set of validation studies (16 total)


# In[31]:


validation_study_set = set()
for study in validation_studies:

    validation_study_set.add(study[0])


# In[ ]:


#Get set of training data


# In[38]:


training_data_list = {"Amyloidosis":[], "HCM":[], "HTN":[]}
validation_data_list = {"Amyloidosis":[], "HCM":[], "HTN":[]}
for key in studydict:

    for study in studydict[key][:-1]:

        if study not in validation_study_set:

            training_data_list[key].append(study)

        else:

            validation_data_list[key].append(study)


# In[ ]:


#split generic training data


# In[39]:


training_data_list_split = {"Amyloidosis":[], "HCM":[], "HTN":[]}

for key in training_data_list:

    while len(training_data_list_split[key]) < 30:

        training_data_list_split[key].append(training_data_list[key].pop())


# In[41]:


print(len(training_data_list['Amyloidosis']), len(training_data_list_split['Amyloidosis']) ,len(validation_data_list['Amyloidosis']))
print(len(training_data_list['HCM']), len(training_data_list_split['Amyloidosis']), len(validation_data_list['HCM']))
print(len(training_data_list['HTN']), len(training_data_list_split['Amyloidosis']), len(validation_data_list['HTN']))


# In[ ]:





# In[ ]:


#Create generic training data


# In[43]:


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


# In[ ]:


#Create view seperated training


# In[22]:


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


# In[23]:


view_file_counter_training


# In[24]:


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


# In[25]:


view_file_counter_validation


# In[ ]:


#Generic Model Creator


# In[30]:


for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

    for disease in ["Amyloidosis", "HCM", "HTN"]:

        if not Path(training_path + "/training_data_generic/{0}/".format(disease)).exists():

            Path(training_path + "/training_data_generic/{0}/".format(disease)).mkdir(parents = True, exist_ok = True)

        for file in Path(training_path + "/training_data/{0}/{1}/".format(view, disease)).glob('**/*'):

            filename = str(file).split('/')[-1]

            shutil.copy(training_path + "/training_data/{0}/{1}/{2}".format(view, disease, filename), training_path + "/training_data_generic/{0}/{1}".format(disease, filename))


# In[ ]:


sum = 0
for study in validation_studies:

    sum += int(study[1])*3*6

print(sum)

#This equals the min nunmber of frames possible for each study * number of studies for the total size of the validation dataset. Value is 16056. Running find . -type f | wc -l in terminal yields 16056. PERFECT
#These images are split into 3 classes: AMY, HCM, HTM and each study may not have the same number of slices; however, total number of class slices are equal. 5352 slices per class.


# In[ ]:


#Get prediction from each model on Validation Data Set
#For each study: 18 by N predictions where N is number of Frames and the 18 predictions are from each view: [AP2[p1,p2,p3], AP3[p1,p2,p3], AP4[p1,p2,p3], PSAX_V[p1,p2,p3], PSAX_M[p1,p2,p3], PLAX[p1,p2,p3]] x 66 Frames (example)
#We have a Ground Truth Label for each study ( 0, 1, 2) (Since each study can only fit into one class)
#Generate Multiclass classifaction fusion model amalgamating the prediction vector to get the per study prediction
#48 studies, 48 fusion model predictions
#NOTE THAT Amyloidosis/R001 != HCM/R001! Studies share names but are distinct, classes do not overlap


# In[ ]:


model_path = r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/models/'


# In[ ]:


mirrored_strategy = tensorflow.distribute.MirroredStrategy()
#mirrored_strategy = tensorflow.distribute.MirroredStrategy(['/gpu:2', '/gpu:3'])


# In[ ]:


with mirrored_strategy.scope():

    loaded_model_AP2 = tensorflow.keras.models.load_model(model_path + "AP2_view_classifier.h5")
    loaded_model_AP3 = tensorflow.keras.models.load_model(model_path + "AP3_view_classifier.h5")
    loaded_model_AP4 = tensorflow.keras.models.load_model(model_path + "AP4_view_classifier.h5")
    loaded_model_PSAX_V = tensorflow.keras.models.load_model(model_path + "PSAX_V_view_classifier.h5")
    loaded_model_PSAX_M = tensorflow.keras.models.load_model(model_path + "PSAX_M_view_classifier.h5")
    loaded_model_PLAX = tensorflow.keras.models.load_model(model_path + "PLAX_view_classifier.h5")


# In[ ]:


from skimage import transform


# In[ ]:


def image_load(filename):
   np_image = PIL.Image.open(filename)
   np_image = numpy.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (299, 299, 3))
   np_image = numpy.expand_dims(np_image, axis=0)
   return np_image

 #image = load('my_file.jpg')
 #model.predict(image)


# In[ ]:


test_image = image_load(r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_validation/min_grouped_data/R010/AP2/Amyloidosis/R010_AP2.avi_9.png')
test_image2 = image_load(r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_validation/min_grouped_data/R010/AP2/Amyloidosis/R010_AP2.avi_10.png')
test_image3 = image_load(r'/home/james/Datacenter_storage/James_ECHO_LVH/new_model/new_model_02/files/study_wise_validation/min_grouped_data/R010/AP2/Amyloidosis/R010_AP2.avi_11.png')


# In[ ]:


image_stack = numpy.vstack((test_image, test_image2, test_image3))


# In[ ]:


print(image_stack.shape)


# In[457]:


AP2_Prediction = loaded_model_AP2.predict(image_stack)


# In[ ]:


print(test_image.shape)


# In[ ]:


print(AP2_Prediction)


# In[ ]:


print(validation_studies)


# In[ ]:


from collections import defaultdict


# In[ ]:


prediction_list = defaultdict(list)
files = []

for disease in ["Amyloidosis", "HCM", "HTN"]:

    for study in validation_studies:

        study_name = study[0] 
        key = "{0}_{1}".format(disease, study_name)
        prediction_list[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

        for view,index in zip(["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"], range(0,6)):

            for file in Path(validation_path + "min_grouped_data/{0}/{1}/{2}/".format(study_name, view, disease)).glob('**/*'):

                files.append(str(file).split('/')[-1])

            files = natsorted(files)
            
            for file in files:

                prediction_list[key][view].append(image_load(validation_path + "min_grouped_data/{0}/{1}/{2}/{3}".format(study_name, view, disease,file)))
            
            files = []

    


# In[198]:


print(prediction_list.keys())


# In[214]:


prediction_output = defaultdict(list)
for key in prediction_list:

    prediction_output[key] = {"AP2":[], "AP3":[], "AP4":[], "PSAX_V":[], "PSAX_M":[], "PLAX":[]}

    for view in ["AP2", "AP3", "AP4", "PSAX_V", "PSAX_M", "PLAX"]:

        image_stack = numpy.vstack(prediction_list[key][view])
        
        if view == "AP2":

            prediction = loaded_model_AP2.predict(image_stack)

        elif view == "AP3":

            prediction = loaded_model_AP3.predict(image_stack)

        elif view == "AP4":

            prediction = loaded_model_AP4.predict(image_stack)

        elif view == "PSAX_V":

            prediction = loaded_model_PSAX_V.predict(image_stack)

        elif view == "PSAX_M":

            prediction = loaded_model_PSAX_M.predict(image_stack)
        
        elif view == "PLAX":

            prediction = loaded_model_PLAX.predict(image_stack)

        prediction_output[key][view].append(prediction)

        print(key)

        


# In[222]:


dataframe_template = {'Amyloidosis Prediction': [], 'HCM Prediction': [], 'HTM Prediction': []}


# In[239]:


prediction_dataframes = {}

for key in prediction_output:

    df_list = []

    for key2 in prediction_output[key]:

        df_list.append(pandas.DataFrame(numpy.array(prediction_output[key][key2][0]), columns = ['Amyloidosis_{0}'.format(key2), 'HCM_{0}'.format(key2), 'HTN_{0}'.format(key2)]))

    result = pandas.concat(df_list, axis=1)
    prediction_dataframes[key] = result
    print(key)


# In[249]:


for key in prediction_dataframes:
    
    prediction_dataframes[key].to_csv("{0}/dataframes/{1}.csv".format(results_path,key))


# In[253]:


fusion_input = {}

for key in prediction_dataframes:
    
    fusion_input[key] = prediction_dataframes[key].to_numpy()


# In[344]:


labels = None
full_data_set = None
counter = 0

for key in fusion_input:
    
    print(key,len(fusion_input[key]))

    if str(key).split('_')[0] == "Amyloidosis":

        val = 0

    elif str(key).split('_')[0] == "HCM":

        val = 1

    elif str(key).split('_')[0] == "HTN":

        val = 2

    
    if counter % 4 == 0:

        if val != 0:

            val -= 1

        elif val != 2:

            val += 1

    counter += 1

    if full_data_set is None:

        full_data_set = fusion_input[key]
        labels = numpy.full((len(fusion_input[key]),1), val, dtype=int)

    else:

        full_data_set = numpy.concatenate((full_data_set, fusion_input[key]))
        labels = numpy.concatenate((labels, numpy.full((len(fusion_input[key]),1), val, dtype=int)))
        


# In[341]:


labels_dict = {}

for key in fusion_input:
    
    print(key,len(fusion_input[key]))

    if str(key).split('_')[0] == "Amyloidosis":

        val = 0

    elif str(key).split('_')[0] == "HCM":

        val = 1

    elif str(key).split('_')[0] == "HTN":

        val = 2

    labels_dict[key] = numpy.full((len(fusion_input[key]),1), val, dtype=int)

        


# In[316]:


full_data_set.shape


# In[317]:


labels.shape


# In[403]:


LR_model = LogisticRegression(solver = "saga", multi_class= 'ovr', C=100, penalty='elasticnet', l1_ratio = 1)
LR_model.fit(full_data_set, labels)


# In[369]:



  
# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
vot_soft.fit(full_data_set, labels)
y_pred = vot_soft.predict(fusion_input['HTN_R155'])
  
# using accuracy_score
score = accuracy_score(labels_dict['HTN_R155'], y_pred)
print(y_pred)
print("Soft Voting Score % d" % score)


# In[429]:


dataframe_template = {'Study':[], 'Amyloidosis': [], 'HCM': [], 'HTN': [], 'Prediction': [], 'Actual': []}
study_wise_predictions_df = pandas.DataFrame(data = dataframe_template)
study_wise_predictions_df


# In[400]:


def get_majority_vote(array):

    col_sum = array.mean(axis=0)
    return col_sum.reshape(1, -1)


# In[430]:


for key in fusion_input:

    if str(key).split('_')[0] == "Amyloidosis":

        val = 0

    elif str(key).split('_')[0] == "HCM":

        val = 1

    elif str(key).split('_')[0] == "HTN":

        val = 2

    to_majority_vote = get_majority_vote(fusion_input[key])
    prediction_prob = LR_model.predict_proba(to_majority_vote)
    prediction = LR_model.predict(to_majority_vote)

    new_row = {'Study': key, 'Amyloidosis': prediction_prob[0][0], 'HCM': prediction_prob[0][1], 'HTN': prediction_prob[0][2], 'Prediction': prediction[0], 'Actual': val}
    study_wise_predictions_df = study_wise_predictions_df.append(new_row, ignore_index=True)
    


# In[435]:


study_wise_predictions_df["Prediction"] = study_wise_predictions_df["Prediction"].astype('int32')
study_wise_predictions_df["Actual"] = study_wise_predictions_df["Actual"].astype('int32')
study_wise_predictions_df.to_csv("{0}/dataframes/{1}.csv".format(results_path,"LogResStudyWise"))
study_wise_predictions_df


# In[449]:


pred_only_df = pandas.concat([study_wise_predictions_df['Amyloidosis'], study_wise_predictions_df['HCM'], study_wise_predictions_df['HTN']], axis=1, keys=['Amyloidosis', 'HCM', 'HTN'])
pred_only_df


# In[446]:


validation_set_df = pandas.DataFrame(0, index = range(len(study_wise_predictions_df)), columns=[0, 1, 2])

for i,j in enumerate(study_wise_predictions_df.iloc[:, 5]):

    validation_set_df[j][i] = 1
    
validation_set_df.rename({0: 'Amyloidosis', 1: 'HCM', 2: 'HTN'}, axis=1, inplace=True)
validation_set_df


# In[447]:


Amyloidosis_AUC_LR = compute_roc(X=study_wise_predictions_df.iloc[:, 1], y=validation_set_df.iloc[:, 0], pos_label=True)
print("Amyloidosis_AUC: ROC-AUC=%.3f" % (Amyloidosis_AUC_LR.auc))
HCM_AUC_LR = compute_roc(X=study_wise_predictions_df.iloc[:, 2], y=validation_set_df.iloc[:, 1], pos_label=True)
print("HCM_AUC: ROC-AUC=%.3f" % (HCM_AUC_LR.auc))
HTN_AUC_LR = compute_roc(X=study_wise_predictions_df.iloc[:, 3], y=validation_set_df.iloc[:, 2], pos_label=True)
print("HTN_AUC: ROC-AUC=%.3f" % (HTN_AUC_LR.auc))


# In[452]:


Y_predicted = []
class_names = ['Amyloidosis', 'HCM', 'HTN']
maxValueIndexObj = pred_only_df.idxmax(axis=1)

for i in maxValueIndexObj:

    Y_predicted.append(pred_only_df.columns.get_loc(i))

con_matrix =  confusion_matrix(study_wise_predictions_df.iloc[:, 5], Y_predicted)
print(con_matrix)

pyplot.figure(figsize = (12, 12))

pyplot.title("Study Based Fusion Confusion Matrix\n")
pyplot.rcParams.update({'font.size': 32})

pyplot.xticks(numpy.arange(len(class_names)), class_names)
pyplot.yticks(numpy.arange(len(class_names)), class_names)


for i, j in itertools.product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):

    if i == j:

        pyplot.text(j, i, con_matrix[i, j], horizontalalignment="center", color="green")

    else:
        
        pyplot.text(j, i, con_matrix[i, j], horizontalalignment="center", color="black")

pyplot.tight_layout()

pyplot.ylabel('Ground Truth')
pyplot.xlabel('\nModel Prediction')

pyplot.imshow (con_matrix, interpolation = 'nearest', cmap = mpl.pyplot.cm.Blues)


# In[455]:



_, ax3 = pyplot.subplots(figsize = (12,12))
plot_roc(Amyloidosis_AUC_LR, label="Amyloidosis", color="red", ax=ax3)
plot_roc(HCM_AUC_LR, label="HCM", color="green", ax=ax3)
plot_roc(HTN_AUC_LR, label="HTN", color="blue", ax=ax3)
# Place the legend outside.
ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax3.set_title("ROC curves")
