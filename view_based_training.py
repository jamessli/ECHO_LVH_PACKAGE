from pathlib import Path

#Tensorflow
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Fusion Models
#Using image data generator, the classes are in the order they appear in the files, but when passed to model, are shuffled. Hence, shuffle needs to be false for the test data set

from sklearn.metrics import confusion_matrix, classification_report
from roc_utils import *
import logging


#The EfficientnetB5 (Great for resolutions around 456 by 456, maybe switch to b7 depending on performance) B4 gives 380 by 380
from tensorflow.keras.applications import InceptionResNetV2

def train_views(training_path, validation_path, output_path, devices):

    mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices)

    training_path_views = training_path
    validation_path_views = validation_path
    model_path = output_path

    batch_size = 32 #4Gpus, 64 batches each?
    epochs = 25
    image_size = 299
    weight_decay = .3
    learnable_layers = .5
    learning_rate = ".0005"
    learning_rate_decay = .5

    if Path('./hyperparameters.txt').is_file():

        with open('./hyperparameters.txt') as in_file:

            lines = in_file.readlines()

            for line in lines:

                if line == "batch_size":

                    batch_size = int(line.split('=')[-1])

                if line == "epochs":

                    epochs = int(line.split('=')[-1])

                if line == "weight_decay":

                    weight_decay = float(line.split('=')[-1])

                if line == "learnable_layers":

                    learnable_layers = float(line.split('=')[-1])

                if line == "learning_rate":

                    learning_rate = float(line.split('=')[-1])

                if line == "learning_rate_decay":

                    learning_rate_decay = float(line.split('=')[-1])


    training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 45, width_shift_range = .5, height_shift_range = .15, horizontal_flip = False, fill_mode = 'nearest') #Rescale images to 0 to .255

    with mirrored_strategy.scope():

        AP2_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        AP3_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        AP4_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        PSAX_V_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        PSAX_M_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")
        PLAX_model = tensorflow.keras.models.load_model(str(model_path) + "/generic_classifier.h5")

    model_list = [AP2_model, AP3_model, AP4_model, PSAX_V_model, PSAX_M_model, PLAX_model]

    for view,model in zip(['AP2', 'AP3', 'AP4', 'PSAX_V', 'PSAX_M', 'PLAX'], model_list):

        training_data_gen = training_data_generator.flow_from_directory(str(training_path_views + '/{0}/'.format(view)),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
        train_len = len(training_data_gen.classes)
        print(view, " training data processed slices: ", training_data_gen.class_indices)

        validation_data_gen = training_data_generator.flow_from_directory(str(validation_path_views + '/{0}/'.format(view)),  shuffle = True, target_size = (image_size,image_size), batch_size = batch_size, color_mode = 'rgb', class_mode = "categorical")
        val_len = len(validation_data_gen.classes)
        print(view, "validation data processed slices: ", validation_data_gen.class_indices)

        make_trainable = round(len(model.layers)*learnable_layers)

        for layer in model.layers[-make_trainable:]:

            layer.trainable = True

            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.kernel))
            
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:

                layer.add_loss(lambda: regularizers.l2(weight_decay)(layer.bias))
                
        reduce_lr  = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = 10, factor = learning_rate_decay, min_lr = .000000001, verbose = 2)

        model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = learning_rate), loss = "categorical_crossentropy", metrics = ["accuracy"])

        history = model.fit(training_data_gen, epochs = epochs, steps_per_epoch = train_len//batch_size, validation_data=validation_data_gen, validation_steps = val_len//batch_size, verbose = 1, callbacks = [reduce_lr], use_multiprocessing = True, workers = 64)
        model.save(model_path + "/{0}_classifier.h5".format(view))

        print(view, " model training accuracy: ", history.history["accuracy"], " | validation accuracy: ", history.history["val_accuracy"])

    return "Finished training view-based classifiers"