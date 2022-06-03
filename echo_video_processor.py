#!/usr/bin/env python
# coding: utf-8

# In[1]:
import cv2
import glob
import numpy
from PIL import Image
from pathlib import Path

from torch import threshold

def split_video(video_path, vid_output_path):

    file_name = str(Path(video_path).name)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    ret,frame = video.read()
    count = 0

    while ret is True:

        cv2.imwrite("{0}/{1}_{2}.png".format(vid_output_path, file_name, count), frame)
        ret,frame = video.read()
        
        count += 1

    return fps

def split_into_frames(video_path, view_output_path):

    file_name = str(Path(video_path).name)
    video = cv2.VideoCapture(video_path)

    ret,frame = video.read()
    count = 0

    while ret is True:
        
        #COLOR CORRECTION STARTS HERE
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        if frame.shape[1] <= 640:

            kernel = numpy.ones((2,1),numpy.uint8)

        elif frame.shape[1] <= 1080:

            kernel = numpy.ones((3,2),numpy.uint8)

        else:

            kernel = numpy.ones((4,4),numpy.uint8)

        lower_yellow = numpy.array([15, 25, 100])
        upper_yellow = numpy.array([45, 255, 220])
        lower_blue = numpy.array([45, 35, 0])
        upper_blue = numpy.array([165, 255, 255])
        low_y = numpy.full((10,10,3), lower_yellow, dtype = numpy.uint8)/255.0
        high_y = numpy.full((10,10,3), upper_yellow, dtype = numpy.uint8)/255.0
        low_b = numpy.full((10,10,3), lower_blue, dtype = numpy.uint8)/255.0
        high_b = numpy.full((10,10,3), upper_blue, dtype = numpy.uint8)/255.0

        mask_y = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)
        mask_y = cv2.bitwise_not(mask_y)

        mask_b = cv2.inRange(frame_hsv, lower_blue, upper_blue)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kernel)
        mask_b = cv2.bitwise_not(mask_b)

        mask_composite = mask_y & mask_b
        mask_composite = cv2.bitwise_not(mask_composite)
        #mask_composite = cv2.morphologyEx(mask_composite, cv2.MORPH_OPEN, kernel)
        mask_composite = cv2.dilate(mask_composite,kernel,iterations = 1)

        frame_colour_corrected = numpy.dstack((frame, mask_composite))
        frame_colour_corrected = Image.fromarray(frame_colour_corrected)
        frame_inpainted = cv2.inpaint(frame, mask_composite, 3, cv2.INPAINT_TELEA)

        #COLOR CORRECTION ENDS HERE

        frame = Image.fromarray(frame_inpainted).convert('L')
        frame = numpy.array(frame)
        #Added
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Added
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        ret2, threshhold = cv2.threshold(blurred, 45, 255, 0)
        #ret3,threshhold = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #threshhold = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        contours, hierarchy = cv2.findContours(threshhold, 1, 2)

        #Approx contour
        cnt = contours[0]
        largest = cv2.contourArea(cnt)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = contours[0]

        #Central points and area
        moments = cv2.moments(cnt)
        cent_x = int(moments['m10']/moments['m00'])
        cent_y = int(moments['m01']/moments['m00'])
        shape_area = cv2.contourArea(cnt)
        shape_perim = cv2.arcLength(cnt, True)

        epsilon = 0.01*shape_perim
        approximation = cv2.approxPolyDP(cnt, epsilon, True)
        convex_hull = cv2.convexHull(cnt)

        contour_mask = numpy.zeros(frame.shape, numpy.uint8)
        contour_mask = cv2.drawContours(contour_mask, [convex_hull], 0, 255, -1)

        frame_output = numpy.dstack((frame, contour_mask))
        frame_output = Image.fromarray(frame_output)
        frame_output = frame_output.resize((299, 299))

        frame_output.save("{0}/{1}_{2}.png".format(view_output_path, file_name, count)) #<--Add to array and save array

        ret,frame = video.read()
        
        count += 1

#Cycle through video directory list

def list_cycler(video_path, study_name, disease_name, view_name, output_path, operation):

    #directory_path = Path("{0}/{1}/{2}".format(study_path, disease_name, study_name))
    output_path = "{0}/{1}/{2}/{3}".format(output_path, disease_name, study_name, view_name)
    original_output_path = "{0}/{1}/{2}/{3}".format('./temp/video_frames', disease_name, study_name, view_name)

    Path.mkdir(Path(output_path), parents=True, exist_ok=True)
    Path.mkdir(Path(original_output_path), parents=True, exist_ok=True)
    
    split_into_frames(video_path, output_path)
    fps = split_video(video_path, original_output_path)

    return view_name, fps
