#!/usr/bin/env python
# coding: utf-8

# In[114]:



import tensorflow as tensorflow
import numpy as numpy
from tensorflow import keras
import matplotlib.cm as cm
import cv2
from natsort import natsorted
import pandas
from pathlib import Path


# In[116]:

study_path = r'./study/'
output_path = r'./temp/'
model_path = r'./models/'
video_path = r'./output'

image_size = (299, 299)
last_conv_layer = "conv_7b"


# In[119]:

def get_image_array(image_path, image_size):

    image = tensorflow.keras.preprocessing.image.load_img(image_path, target_size = image_size)
    array = tensorflow.keras.preprocessing.image.img_to_array(image)
    array = numpy.expand_dims(array, axis = 0)

    return array


# In[121]:

def make_heatmap(image_array, model, last_conv_layer, pred_index = None):

    grad_model = tensorflow.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])

    with tensorflow.GradientTape() as tape:

        last_conv_layer_output, preds = grad_model(image_array)

        if pred_index is None:

            pred_index = tensorflow.argmax(preds[0])

        class_channel = preds[:, pred_index]

        gradients = tape.gradient(class_channel, last_conv_layer_output)
        pooled_gradients = tensorflow.reduce_mean(gradients, axis = (0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        
        heatmap = last_conv_layer_output @ pooled_gradients[..., tensorflow.newaxis]
        heatmap = tensorflow.squeeze(heatmap)
        heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)

        heatmap =  heatmap.numpy()
        return heatmap


# In[143]:

def save_and_display_gradcam(image_path, study_name, view, heatmap, saved_name, prediction_string, alpha = 0.25):

    image_array = tensorflow.keras.preprocessing.image.load_img(image_path)
    image_array = tensorflow.keras.preprocessing.image.img_to_array(image_array)

    heatmap = numpy.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, dsize=(image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    jet = cm.get_cmap("jet")
    
    jet_colors = jet(numpy.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_array.shape[1], image_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    jet_heatmap[:,:,2] = numpy.zeros([jet_heatmap.shape[0], jet_heatmap.shape[1]])

    superimposed = jet_heatmap * alpha + image_array

    cv2.putText(img = superimposed, text = prediction_string, org = (image_array.shape[1]//5, image_array.shape[0]//5), fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = .6, color = (255,0,0), thickness= 1)

    superimposed = keras.preprocessing.image.array_to_img(superimposed)

    save_path = "{0}/{1}_overlays/{2}/{3}".format(output_path,study_name, view, saved_name)
    superimposed.save(save_path)


# In[148]:
def load_study(study_name, view, model, overlay_model, fps):

    raw_frames_path = "{0}/{1}_raw_frames".format(output_path, study_name)
    vid_frames_path = "{0}/{1}_vid_frames".format(output_path, study_name)
    overlay_output_path = "{0}/{1}_overlays/{2}/".format(output_path, study_name, view)
    dataframe_output_path = "{0}/{1}_dataframes/".format(output_path, study_name)

    raw_files = {}
    vid_files = {}

    Path(overlay_output_path).mkdir(parents=True, exist_ok=True)
    Path(dataframe_output_path).mkdir(parents=True, exist_ok=True)

    image_list = list(Path(raw_frames_path + "/{0}/".format(view)).glob('**/*.png'))
    video_list = list(Path(vid_frames_path + "/{0}/".format(view)).glob('**/*.png'))
    #dataframes[study] = pandas.read_csv(dataframe_path + "/{0}/R{1}/{2}/test.csv".format(disease, study_name, view)).iloc[:, 1:]

    temp = []

    for image_ in image_list:

        img = str(image_)
        dir = img.rsplit("/", 1)[0]
        temp.append(img.split("/")[-1])

    temp = natsorted(temp) 

    for j in range(len(temp)):

        temp[j] = dir + "/" + temp[j]

    image_list = temp

    temp = []

    for img_path in image_list:

        temp.append(get_image_array(img_path, image_size = image_size))

    raw_files[study_name] = temp #Sort the processed_files

    temp = []

    for image_ in video_list:

        img = str(image_)
        dir = img.rsplit("/", 1)[0]
        temp.append(img.split("/")[-1])

    temp = natsorted(temp)

    for j in range(len(temp)):

        temp[j] = dir + "/" + temp[j]

    video_list = temp
    vid_files[study_name] = temp #Sort the original files

    for study in raw_files:

        counter = 0
        predictions = []
        predictions_argmax = []

        for img in raw_files[study]:

            prediction = model.predict(img)
            prediction_argmax = numpy.argmax(prediction)

            predictions.append(prediction)
            predictions_argmax.append(prediction_argmax)

            heatmap = make_heatmap(img, overlay_model, last_conv_layer)

            try:

                prediction_string = "AMY: {0}  HCM: {1}  HTN: {2}  ".format(prediction[0][0].round(3), prediction[0][1].round(3), prediction[0][2].round(3))

            except:

                prediction_string = ""
            
            for j in range(len(heatmap)):

                for k in range(len(heatmap[0])):
                    
                    if heatmap[j][k] < .8:

                        heatmap[j][k] = 0
            
            save_file_name = "{0}_".format(study_name) + str(counter) + ".png"
            save_and_display_gradcam(vid_files[study_name][counter], study_name, view, heatmap, save_file_name, prediction_string)

            counter += 1
        '''
        probabilities = pandas.DataFrame(numpy.asarray(predictions), columns = ['HTN','HCM','Amyloidosis'])
        probabilities_argmax = pandas.DataFrame(numpy.asarray(predictions_argmax), columns = ['HTN','HCM','Amyloidosis'])
        probabilities.to_csv("{0}/{2}_probabilities.csv".format(dataframe_output_path, view))
        probabilities_argmax.to_csv("{0}/{2}_argmax.csv".format(dataframe_output_path, view))
        '''
    for study in raw_files:

        to_vid = list(Path(overlay_output_path).glob('**/*.png'))

        temp = []

        for image_ in to_vid:

            img = str(image_)
            dir = img.rsplit("/", 1)[0]
            temp.append(img.split("/")[-1])

        temp = natsorted(temp)

        for j in range(len(to_vid)):

            to_vid[j] = dir + "/" + temp[j]

        img_arr = []
        for filepath in to_vid:

            file = cv2.imread(filepath)
            height, width, _ = file.shape
            size = (width, height)
            img_arr.append(file)

        out = cv2.VideoWriter(video_path + "/{0}/{1}_{2}.mp4".format(study_name, study_name, view), cv2.VideoWriter_fourcc(*'H264'),fps, size)

        for i in range(len(img_arr)):

            out.write(img_arr[i])

        out.release()